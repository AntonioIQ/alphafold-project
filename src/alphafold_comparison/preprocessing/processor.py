#!/usr/bin/env python3
"""
Pipeline de procesamiento de estructuras con mapeo SIFTS correcto.

Objetivos:
1. MAPEO SIFTS: Identificar cadena correcta + mapeo PDB→UniProt de residuos
2. EXTRACCIÓN: CAs de la cadena correcta con posiciones mapeadas
3. SUPERPOSICIÓN: Alineación Kabsch sobre residuos correspondientes
"""

import argparse
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB import Superimposer
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from alphafold_comparison.config import Config

warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)


def _build_sifts_index(sifts_file):
    """
    Construye índice SIFTS: (PDB_upper, SP) -> [(chain, pdb_beg, pdb_end, sp_beg, sp_end)].

    Maneja casos donde PDB_BEG/PDB_END son null:
    - Si PDB_BEG null y SP_BEG == 1: asume PDB_BEG=1 (numeración idéntica)
    - Si PDB_BEG null y SP_BEG != 1: marca con flag para fallback
    - Si PDB_BEG tiene insertion code (ej: "1H"): usa solo parte numérica
    """
    sifts = pd.read_csv(sifts_file, comment='#', low_memory=False)
    sifts.columns = sifts.columns.str.strip()

    index = defaultdict(list)
    for _, row in sifts.iterrows():
        try:
            pdb_upper = str(row['PDB']).upper().strip()
            sp = str(row['SP_PRIMARY']).strip()
            chain = str(row['CHAIN']).strip()
            sp_beg = int(row['SP_BEG'])
            sp_end = int(row['SP_END'])

            pdb_beg_raw = row['PDB_BEG']
            pdb_end_raw = row['PDB_END']

            # Intentar parsear PDB_BEG
            pdb_beg = None
            if pd.notna(pdb_beg_raw):
                try:
                    pdb_beg = int(pdb_beg_raw)
                except (ValueError, TypeError):
                    # Insertion code como "1H" → extraer parte numérica
                    import re
                    m = re.match(r'^(-?\d+)', str(pdb_beg_raw).strip())
                    if m:
                        pdb_beg = int(m.group(1))

            # Intentar parsear PDB_END
            pdb_end = None
            if pd.notna(pdb_end_raw):
                try:
                    pdb_end = int(pdb_end_raw)
                except (ValueError, TypeError):
                    import re
                    m = re.match(r'^(-?\d+)', str(pdb_end_raw).strip())
                    if m:
                        pdb_end = int(m.group(1))

            # Fallback: si no pudimos parsear PDB_BEG/END, asumir misma numeración
            if pdb_beg is None:
                pdb_beg = sp_beg
            if pdb_end is None:
                pdb_end = pdb_beg + (sp_end - sp_beg)

            index[(pdb_upper, sp)].append((chain, pdb_beg, pdb_end, sp_beg, sp_end))
        except (ValueError, TypeError):
            continue

    return dict(index)


def _get_sifts_mapping(sifts_index, pdb_id, uniprot_id):
    """
    Obtiene cadena correcta y mapeo PDB_pos→UniProt_pos.

    Returns:
        (chain_id, {pdb_resnum: uniprot_resnum}) o (None, None) si no hay mapeo.
    """
    key = (pdb_id.upper().strip(), uniprot_id.strip())
    entries = sifts_index.get(key)
    if not entries:
        return None, None

    chain_id = entries[0][0]
    pdb_to_uniprot = {}

    for ch, pb, pe, sb, se in entries:
        if ch != chain_id:
            continue
        pdb_range = list(range(pb, pe + 1))
        sp_range = list(range(sb, se + 1))
        if len(pdb_range) == len(sp_range):
            for p, s in zip(pdb_range, sp_range):
                pdb_to_uniprot[p] = s

    return chain_id, pdb_to_uniprot


# Variables globales para multiprocessing
_global_sifts_index = None
_global_parser = None


def _init_worker(sifts_index):
    """Inicializa variables globales en cada worker."""
    global _global_sifts_index, _global_parser
    _global_sifts_index = sifts_index
    _global_parser = PDB.PDBParser(QUIET=True, PERMISSIVE=True)


def _process_pair_worker(args):
    """Worker function para ProcessPoolExecutor."""
    pdb_id, uniprot_id, min_residues, min_atoms_per_res, min_alignment_atoms = args

    result = {
        "pdb_id": pdb_id,
        "uniprot_id": uniprot_id,
        "success": False,
        "error": None,
        "chain_id": "",
        "matched_residues": 0,
        "rmsd": 0.0,
        "pdb_residues": 0,
        "af_residues": 0,
    }

    try:
        # 1. MAPEO SIFTS
        chain_id, pdb_to_uniprot = _get_sifts_mapping(
            _global_sifts_index, pdb_id, uniprot_id
        )
        if chain_id is None:
            result["error"] = "Sin mapeo SIFTS"
            return result
        if not pdb_to_uniprot:
            result["error"] = "Mapeo SIFTS vacio"
            return result

        result["chain_id"] = chain_id

        # 2. ARCHIVOS
        pdb_file = Config.PDB_DIR / f"pdb{pdb_id.lower()}.ent"
        af_file = Config.AF_DIR / f"{uniprot_id}.pdb"

        if not pdb_file.exists():
            result["error"] = f"PDB no encontrado: {pdb_file.name}"
            return result
        if not af_file.exists():
            result["error"] = f"AlphaFold no encontrado: {af_file.name}"
            return result
        if pdb_file.stat().st_size < 200:
            result["error"] = "Archivo PDB demasiado pequeno"
            return result
        if af_file.stat().st_size < 200:
            result["error"] = "Archivo AlphaFold demasiado pequeno"
            return result

        # 3. PARSING
        pdb_structure = _global_parser.get_structure("pdb", str(pdb_file))
        af_structure = _global_parser.get_structure("af", str(af_file))

        # 4. EXTRAER CAs con mapeo SIFTS
        # PDB: solo cadena correcta, posiciones mapeadas a UniProt
        pdb_cas = {}
        for model in pdb_structure:
            for chain in model:
                if chain.id == chain_id:
                    for res in chain:
                        if res.id[0] == ' ':
                            pdb_pos = res.id[1]
                            uniprot_pos = pdb_to_uniprot.get(pdb_pos)
                            if uniprot_pos is not None:
                                for atom in res:
                                    if atom.name == 'CA':
                                        pdb_cas[uniprot_pos] = atom
            break  # Solo primer modelo

        # AlphaFold: numeración UniProt directamente
        af_cas = {}
        for model in af_structure:
            for chain in model:
                for res in chain:
                    if res.id[0] == ' ':
                        for atom in res:
                            if atom.name == 'CA':
                                af_cas[res.id[1]] = atom
            break

        result["pdb_residues"] = len(pdb_cas)
        result["af_residues"] = len(af_cas)

        # 5. POSICIONES COMUNES (numeración UniProt)
        common_positions = sorted(set(pdb_cas.keys()) & set(af_cas.keys()))
        if len(common_positions) < min_residues:
            result["error"] = f"Insuficientes residuos comunes: {len(common_positions)}"
            return result

        # 6. SUPERPOSICIÓN KABSCH
        pdb_atoms = [pdb_cas[pos] for pos in common_positions]
        af_atoms = [af_cas[pos] for pos in common_positions]

        superimposer = Superimposer()
        superimposer.set_atoms(pdb_atoms, af_atoms)

        result.update({
            "success": True,
            "matched_residues": len(common_positions),
            "rmsd": superimposer.rms,
        })

    except Exception as e:
        result["error"] = str(e)

    return result


class QualityProcessor:
    """
    Procesador de estructuras con mapeo SIFTS y validación de calidad.

    Usa SIFTS para:
    - Identificar la cadena correcta del PDB para cada UniProt ID
    - Mapear numeración de residuos PDB → UniProt (= AlphaFold)
    - Superponer solo los residuos que realmente corresponden
    """

    def __init__(self, min_residues=None, min_atoms_per_res=None, min_alignment_atoms=None):
        self.min_residues = min_residues or Config.MIN_RESIDUES
        self.min_atoms_per_res = min_atoms_per_res or Config.MIN_ATOMS_PER_RES
        self.min_alignment_atoms = min_alignment_atoms or Config.MIN_ALIGNMENT_ATOMS
        self.stats = {"total": 0, "success": 0, "failed": 0}
        self.sifts_index = None

    def _load_sifts(self):
        """Carga índice SIFTS una sola vez."""
        if self.sifts_index is None:
            print(f"Cargando indice SIFTS desde {Config.SIFTS_FILE}...")
            t0 = time.time()
            self.sifts_index = _build_sifts_index(str(Config.SIFTS_FILE))
            print(f"  {len(self.sifts_index):,} pares indexados ({time.time()-t0:.1f}s)")
        return self.sifts_index

    def process_all(self, pairs_df: pd.DataFrame, max_workers=None) -> list:
        """Procesa todos los pares con SIFTS mapping y multiprocessing."""
        max_workers = max_workers or Config.DEFAULT_WORKERS
        self.stats["total"] = len(pairs_df)

        sifts_index = self._load_sifts()

        print(f"Procesando {len(pairs_df):,} pares con mapeo SIFTS")
        print(f"  Workers: {max_workers}")
        print(f"  Min residuos: {self.min_residues}")

        args_list = [
            (row["PDB"], row["SP_PRIMARY"],
             self.min_residues, self.min_atoms_per_res, self.min_alignment_atoms)
            for _, row in pairs_df.iterrows()
        ]

        results = []
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(sifts_index,)
        ) as executor:
            for result in tqdm(
                executor.map(_process_pair_worker, args_list, chunksize=50),
                total=len(args_list),
                desc="Procesando",
                unit="pares",
            ):
                results.append(result)

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        self.stats.update({"success": len(successful), "failed": len(failed)})

        self._generate_report(results)
        return results

    def _generate_report(self, results: list):
        """Genera reportes de procesamiento."""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        print(f"\n{'=' * 60}")
        print(f"RESULTADOS DE PROCESAMIENTO")
        print(f"{'=' * 60}")
        print(f"Total: {len(results):,}")
        print(f"Exito: {len(successful):,} ({len(successful) / len(results) * 100:.1f}%)")
        print(f"Fallos: {len(failed):,} ({len(failed) / len(results) * 100:.1f}%)")

        if successful:
            rmsds = [r["rmsd"] for r in successful]
            matched = [r["matched_residues"] for r in successful]
            print(f"\nMETRICAS:")
            print(f"  Residuos mapeados promedio: {np.mean(matched):.1f}")
            print(f"  RMSD promedio: {np.mean(rmsds):.2f} A")
            print(f"  RMSD mediana: {np.median(rmsds):.2f} A")
            print(f"  RMSD rango: {np.min(rmsds):.2f} - {np.max(rmsds):.2f} A")
            print(f"  < 1A: {sum(1 for r in rmsds if r < 1):,} ({sum(1 for r in rmsds if r < 1)/len(rmsds)*100:.1f}%)")
            print(f"  < 2A: {sum(1 for r in rmsds if r < 2):,} ({sum(1 for r in rmsds if r < 2)/len(rmsds)*100:.1f}%)")
            print(f"  > 5A: {sum(1 for r in rmsds if r > 5):,} ({sum(1 for r in rmsds if r > 5)/len(rmsds)*100:.1f}%)")

        if failed:
            print(f"\nANALISIS DE ERRORES:")
            error_counts = {}
            for r in failed:
                error_type = r["error"].split(":")[0] if r["error"] else "Unknown"
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {error_type}: {count}")

        # Guardar reporte CSV
        Config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_data = [{
            "pdb_id": r["pdb_id"],
            "uniprot_id": r["uniprot_id"],
            "chain_id": r["chain_id"],
            "success": r["success"],
            "error": r["error"] or "",
            "matched_residues": r["matched_residues"],
            "rmsd": r["rmsd"],
            "pdb_residues": r["pdb_residues"],
            "af_residues": r["af_residues"],
        } for r in results]

        df = pd.DataFrame(report_data)
        report_path = Config.REPORTS_DIR / "quality_processing_report.csv"
        df.to_csv(report_path, index=False)
        print(f"Reporte: {report_path}")

        # Guardar índice de exitosos
        if successful:
            success_data = [{
                "pair_id": f"{r['pdb_id']}_{r['uniprot_id']}",
                "pdb_id": r["pdb_id"],
                "uniprot_id": r["uniprot_id"],
                "chain_id": r["chain_id"],
                "matched_residues": r["matched_residues"],
                "rmsd": r["rmsd"],
                "quality_certified": True,
            } for r in successful]

            success_df = pd.DataFrame(success_data)
            success_df.to_csv(Config.QUALITY_INDEX, index=False)
            print(f"Indice: {Config.QUALITY_INDEX}")
            print(f"  UniProt IDs unicos: {success_df['uniprot_id'].nunique()}")


def run_process(workers=None, min_residues=None, min_atoms=None):
    """Ejecuta el pipeline de procesamiento completo."""
    Config.ensure_dirs()

    print("=" * 60)
    print("QUALITY STRUCTURE PROCESSOR v7.0 (SIFTS)")
    print("=" * 60)
    print("Pipeline:")
    print("  1. MAPEO SIFTS: Cadena correcta + PDB->UniProt residues")
    print("  2. EXTRACCION: CAs de cadena mapeada")
    print("  3. SUPERPOSICION: Kabsch sobre residuos correspondientes")
    print("=" * 60)

    if not Config.SIFTS_FILE.exists():
        raise FileNotFoundError(f"SIFTS no encontrado: {Config.SIFTS_FILE}")

    if not Config.ID_LIST.exists():
        raise FileNotFoundError(f"No se encontro: {Config.ID_LIST}")

    pairs_df = pd.read_csv(Config.ID_LIST)
    print(f"Cargados {len(pairs_df):,} pares")

    if "PDB" not in pairs_df.columns or "SP_PRIMARY" not in pairs_df.columns:
        raise ValueError("Columnas requeridas: PDB, SP_PRIMARY")

    processor = QualityProcessor(
        min_residues=min_residues,
        min_atoms_per_res=min_atoms,
    )
    results = processor.process_all(pairs_df, max_workers=workers)

    successful = [r for r in results if r["success"]]
    success_rate = len(successful) / len(pairs_df) * 100

    print(f"\n{'=' * 60}")
    print(f"PROCESAMIENTO COMPLETADO")
    print(f"{'=' * 60}")
    print(f"Resultados certificados: {len(successful):,}/{len(pairs_df):,} ({success_rate:.1f}%)")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Procesa estructuras PDB vs AlphaFold con mapeo SIFTS"
    )
    parser.add_argument("--workers", type=int, default=Config.DEFAULT_WORKERS)
    parser.add_argument("--min-residues", type=int, default=Config.MIN_RESIDUES)
    parser.add_argument("--min-atoms", type=int, default=Config.MIN_ATOMS_PER_RES)
    args = parser.parse_args()

    return run_process(
        workers=args.workers,
        min_residues=args.min_residues,
        min_atoms=args.min_atoms,
    )


if __name__ == "__main__":
    sys.exit(main())
