"""
Análisis a nivel atómico de estructuras PDB vs AlphaFold.

Usa mapeo SIFTS para comparar correctamente por tipo de átomo y elemento (CHONSP).
"""

import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from alphafold_comparison.config import Config
from alphafold_comparison.preprocessing.processor import _build_sifts_index, _get_sifts_mapping

warnings.filterwarnings("ignore")

_parser = PDBParser(QUIET=True)
_sifts_index = None

PROTEIN_ELEMENTS = {"C", "H", "O", "N", "S", "P"}


def _init_atomic_worker(sifts_index):
    """Inicializa SIFTS index en cada worker."""
    global _sifts_index
    _sifts_index = sifts_index


def _get_element(atom):
    """Extrae el símbolo de elemento de un átomo BioPython."""
    if hasattr(atom, "element") and atom.element.strip():
        return atom.element.strip().upper()
    name = atom.get_name().strip()
    if name:
        return name[0].upper()
    return "?"


def analyze_pair_atomic(row: dict) -> dict:
    """Analiza un par a nivel atómico con mapeo SIFTS."""
    pdb_id = row["pdb_id"]
    uniprot_id = row["uniprot_id"]
    pdb_dir = row.get("pdb_dir", str(Config.PDB_DIR))
    af_dir = row.get("af_dir", str(Config.AF_DIR))

    # Mapeo SIFTS
    chain_id, pdb_to_uniprot = _get_sifts_mapping(_sifts_index, pdb_id, uniprot_id)
    if chain_id is None or not pdb_to_uniprot:
        return None

    from pathlib import Path
    pdb_file = Path(pdb_dir) / f"pdb{pdb_id.lower()}.ent"
    af_file = Path(af_dir) / f"{uniprot_id}.pdb"

    if not pdb_file.exists() or not af_file.exists():
        return None

    try:
        pdb_struct = _parser.get_structure("pdb", str(pdb_file))
        af_struct = _parser.get_structure("af", str(af_file))

        # PDB: cadena SIFTS con posiciones mapeadas a UniProt
        pdb_residues = {}
        for model in pdb_struct:
            for chain in model:
                if chain.id == chain_id:
                    for res in chain:
                        if res.id[0] == " " and "CA" in res:
                            pdb_pos = res.id[1]
                            uniprot_pos = pdb_to_uniprot.get(pdb_pos)
                            if uniprot_pos is not None:
                                pdb_residues[uniprot_pos] = res
            break

        # AlphaFold: numeración UniProt directa
        af_residues = {}
        for model in af_struct:
            for chain in model:
                for res in chain:
                    if res.id[0] == " " and "CA" in res:
                        af_residues[res.id[1]] = res
            break

        common_ids = sorted(set(pdb_residues.keys()) & set(af_residues.keys()))
        if len(common_ids) < Config.MIN_COMMON_RESIDUES:
            return None

        # Alinear por CA
        pdb_cas = [pdb_residues[i]["CA"] for i in common_ids]
        af_cas = [af_residues[i]["CA"] for i in common_ids]

        sup = Superimposer()
        sup.set_atoms(pdb_cas, af_cas)
        sup.apply(af_struct.get_atoms())

        # Analizar por elemento
        element_dists = defaultdict(list)
        backbone_names = {"N", "CA", "C", "O"}
        backbone_dists = []
        sidechain_dists = []

        for res_id in common_ids:
            pdb_res = pdb_residues[res_id]
            af_res = af_residues[res_id]

            pdb_atoms = {a.get_name(): a for a in pdb_res}
            af_atoms = {a.get_name(): a for a in af_res}

            for atom_name in set(pdb_atoms.keys()) & set(af_atoms.keys()):
                element = _get_element(pdb_atoms[atom_name])
                dist = np.linalg.norm(
                    pdb_atoms[atom_name].get_coord() - af_atoms[atom_name].get_coord()
                )
                element_dists[element].append(dist)

                if atom_name in backbone_names:
                    backbone_dists.append(dist)
                else:
                    sidechain_dists.append(dist)

        result = {
            "pdb_id": pdb_id,
            "uniprot_id": uniprot_id,
            "chain_id": chain_id,
            "protein_length": len(common_ids),
            "global_rmsd": sup.rms,
        }

        for element in PROTEIN_ELEMENTS:
            if element in element_dists:
                dists = np.array(element_dists[element])
                result[f"{element}_count"] = len(dists)
                result[f"{element}_mean_rmsd"] = float(np.mean(dists))
                result[f"{element}_median_rmsd"] = float(np.median(dists))
                result[f"{element}_std_rmsd"] = float(np.std(dists))
            else:
                result[f"{element}_count"] = 0
                result[f"{element}_mean_rmsd"] = np.nan
                result[f"{element}_median_rmsd"] = np.nan
                result[f"{element}_std_rmsd"] = np.nan

        if backbone_dists:
            result["backbone_mean_rmsd"] = float(np.mean(backbone_dists))
        if sidechain_dists:
            result["sidechain_mean_rmsd"] = float(np.mean(sidechain_dists))

        return result

    except Exception:
        return None


def run_atomic_analysis(max_pairs=None, workers=None):
    """Ejecuta análisis atómico completo con mapeo SIFTS."""
    workers = workers or Config.DEFAULT_WORKERS

    print("=" * 60)
    print("ATOMIC ANALYSIS v2.0 (SIFTS)")
    print("=" * 60)

    # Cargar SIFTS
    print("Cargando indice SIFTS...")
    sifts_index = _build_sifts_index(str(Config.SIFTS_FILE))
    print(f"  {len(sifts_index):,} pares indexados")

    if Config.QUALITY_INDEX.exists():
        pairs_df = pd.read_csv(Config.QUALITY_INDEX)
        pdb_col, uniprot_col = "pdb_id", "uniprot_id"
    else:
        pairs_df = pd.read_csv(Config.ID_LIST)
        pdb_col, uniprot_col = "PDB", "SP_PRIMARY"

    if max_pairs:
        pairs_df = pairs_df.head(max_pairs)

    rows = [{
        "pdb_id": row[pdb_col],
        "uniprot_id": row[uniprot_col],
        "pdb_dir": str(Config.PDB_DIR),
        "af_dir": str(Config.AF_DIR),
    } for _, row in pairs_df.iterrows()]

    print(f"Analizando {len(rows):,} pares a nivel atomico...")

    results = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_atomic_worker,
        initargs=(sifts_index,)
    ) as executor:
        for result in tqdm(
            executor.map(analyze_pair_atomic, rows, chunksize=50),
            total=len(rows), desc="Atomico", unit="pares"
        ):
            if result is not None:
                results.append(result)

    if not results:
        print("Sin resultados")
        return None

    results_df = pd.DataFrame(results)
    output = Config.RESULTS_DIR / "atomic_analysis_results.csv"
    results_df.to_csv(output, index=False)

    print(f"\nResultados: {len(results_df):,} pares analizados")
    print(f"Guardado: {output}")

    print(f"\nRMSD medio por elemento:")
    for element in sorted(PROTEIN_ELEMENTS):
        col = f"{element}_mean_rmsd"
        if col in results_df.columns:
            mean_val = results_df[col].dropna().mean()
            count_col = f"{element}_count"
            total = results_df[count_col].sum()
            print(f"  {element}: {mean_val:.3f} A (n={total:,.0f})")

    return results_df
