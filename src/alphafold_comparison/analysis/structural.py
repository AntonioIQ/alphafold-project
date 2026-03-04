#!/usr/bin/env python3
"""
Análisis estadístico de pares de estructuras PDB vs AlphaFold.

Usa mapeo SIFTS para comparar correctamente los residuos correspondientes.
Calcula métricas de RMSD y genera reportes estadísticos.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm

from alphafold_comparison.config import Config
from alphafold_comparison.preprocessing.processor import _build_sifts_index, _get_sifts_mapping

warnings.filterwarnings("ignore")

_parser = PDBParser(QUIET=True)
_sifts_index = None


def _init_analysis_worker(sifts_index):
    """Inicializa SIFTS index en cada worker."""
    global _sifts_index
    _sifts_index = sifts_index


def analyze_structure_pair(row: dict) -> dict:
    """
    Analiza un par de estructuras PDB y AlphaFold con mapeo SIFTS.

    Args:
        row: dict con pdb_id, uniprot_id, pdb_dir, af_dir.

    Returns:
        dict con métricas o None si falla.
    """
    pdb_id = row.get("pdb_id") or row.get("PDB")
    uniprot_id = row.get("uniprot_id") or row.get("SP_PRIMARY")
    pdb_dir = row.get("pdb_dir")
    af_dir = row.get("af_dir")

    if not pdb_id or not uniprot_id:
        return None

    # Obtener mapeo SIFTS
    chain_id, pdb_to_uniprot = _get_sifts_mapping(_sifts_index, pdb_id, uniprot_id)
    if chain_id is None or not pdb_to_uniprot:
        return None

    pdb_file = Path(pdb_dir) / f"pdb{pdb_id.lower()}.ent"
    af_file = Path(af_dir) / f"{uniprot_id}.pdb"

    if not pdb_file.exists() or not af_file.exists():
        return None

    try:
        pdb_structure = _parser.get_structure("pdb", str(pdb_file))
        af_structure = _parser.get_structure("af", str(af_file))

        # Extraer CAs del PDB (cadena SIFTS, posiciones mapeadas a UniProt)
        pdb_cas = {}
        for model in pdb_structure:
            for chain in model:
                if chain.id == chain_id:
                    for res in chain:
                        if res.id[0] == " " and "CA" in res:
                            pdb_pos = res.id[1]
                            uniprot_pos = pdb_to_uniprot.get(pdb_pos)
                            if uniprot_pos is not None:
                                pdb_cas[uniprot_pos] = res["CA"]
            break

        # Extraer CAs de AlphaFold (numeración UniProt directa)
        af_cas = {}
        for model in af_structure:
            for chain in model:
                for res in chain:
                    if res.id[0] == " " and "CA" in res:
                        af_cas[res.id[1]] = res["CA"]
            break

        common_res_ids = sorted(set(pdb_cas.keys()) & set(af_cas.keys()))
        if len(common_res_ids) < Config.MIN_COMMON_RESIDUES:
            return None

        pdb_cas_filtered = [pdb_cas[rid] for rid in common_res_ids]
        af_cas_filtered = [af_cas[rid] for rid in common_res_ids]

        # Superposición Kabsch
        super_imposer = Superimposer()
        super_imposer.set_atoms(pdb_cas_filtered, af_cas_filtered)
        super_imposer.apply(af_cas_filtered)

        # RMSD por residuo
        rmsd_per_residue = []
        for pdb_ca, af_ca in zip(pdb_cas_filtered, af_cas_filtered):
            dist = np.linalg.norm(pdb_ca.get_coord() - af_ca.get_coord())
            rmsd_per_residue.append(dist)

        rmsd_array = np.array(rmsd_per_residue)

        return {
            "pdb_id": pdb_id,
            "uniprot_id": uniprot_id,
            "chain_id": chain_id,
            "protein_length": len(common_res_ids),
            "global_rmsd": super_imposer.rms,
            "mean_rmsd": np.mean(rmsd_array),
            "std_rmsd": np.std(rmsd_array),
            "median_rmsd": np.median(rmsd_array),
            "max_rmsd": np.max(rmsd_array),
            "min_rmsd": np.min(rmsd_array),
            "high_rmsd_count": int(np.sum(rmsd_array > 2.0)),
            "percent_high_rmsd": float((np.sum(rmsd_array > 2.0) / len(rmsd_array)) * 100),
            "q25_rmsd": float(np.percentile(rmsd_array, 25)),
            "q75_rmsd": float(np.percentile(rmsd_array, 75)),
        }

    except Exception:
        return None


def generate_statistics_report(results_df: pd.DataFrame, output_dir: Path):
    """Genera reporte de estadísticas descriptivas."""
    report_path = output_dir / "summary_statistics.txt"

    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ANALISIS ESTADISTICO: PDB vs AlphaFold (SIFTS corregido)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total de pares analizados: {len(results_df):,}\n")
        f.write(f"UniProt IDs unicos: {results_df['uniprot_id'].nunique()}\n\n")

        f.write("RMSD GLOBAL:\n")
        f.write("-" * 40 + "\n")
        for stat, value in results_df["global_rmsd"].describe().items():
            f.write(f"  {stat:12}: {value:.4f} A\n")

        f.write(f"\n  < 0.5A: {(results_df['global_rmsd'] < 0.5).sum():,} "
                f"({(results_df['global_rmsd'] < 0.5).mean()*100:.1f}%)\n")
        f.write(f"  < 1.0A: {(results_df['global_rmsd'] < 1.0).sum():,} "
                f"({(results_df['global_rmsd'] < 1.0).mean()*100:.1f}%)\n")
        f.write(f"  < 2.0A: {(results_df['global_rmsd'] < 2.0).sum():,} "
                f"({(results_df['global_rmsd'] < 2.0).mean()*100:.1f}%)\n")
        f.write(f"  > 5.0A: {(results_df['global_rmsd'] > 5.0).sum():,} "
                f"({(results_df['global_rmsd'] > 5.0).mean()*100:.1f}%)\n")

        f.write("\nDISTRIBUCION POR TAMANO DE PROTEINA:\n")
        f.write("-" * 40 + "\n")
        size_bins = [0, 100, 200, 300, 400, 500, 1000, float("inf")]
        size_labels = ["<100", "100-200", "200-300", "300-400", "400-500", "500-1000", ">1000"]
        results_df["size_category"] = pd.cut(
            results_df["protein_length"], bins=size_bins, labels=size_labels
        )
        for cat in size_labels:
            subset = results_df[results_df["size_category"] == cat]
            if len(subset) > 0:
                f.write(
                    f"  {cat:10}: {len(subset):5} pares, "
                    f"RMSD medio: {subset['global_rmsd'].mean():.2f} A, "
                    f"mediana: {subset['global_rmsd'].median():.2f} A, "
                    f"<2A: {(subset['global_rmsd'] < 2).mean()*100:.1f}%\n"
                )

    print(f"Estadisticas guardadas: {report_path}")


def run_analysis(max_pairs=None, workers=None):
    """Ejecuta el pipeline de análisis completo con mapeo SIFTS."""
    workers = workers or Config.DEFAULT_WORKERS
    Config.ensure_dirs()

    output_dir = Config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STRUCTURE COMPARISON ANALYSIS v3.0 (SIFTS)")
    print("=" * 60)

    # Cargar SIFTS
    if not Config.SIFTS_FILE.exists():
        raise FileNotFoundError(f"SIFTS no encontrado: {Config.SIFTS_FILE}")

    print("Cargando indice SIFTS...")
    sifts_index = _build_sifts_index(str(Config.SIFTS_FILE))
    print(f"  {len(sifts_index):,} pares indexados")

    # Cargar pares
    pairs_file = Config.QUALITY_INDEX if Config.QUALITY_INDEX.exists() else Config.ID_LIST
    if not pairs_file.exists():
        raise FileNotFoundError(f"No se encontro archivo de pares: {pairs_file}")

    pairs_df = pd.read_csv(pairs_file)
    print(f"Cargados {len(pairs_df):,} pares desde {pairs_file.name}")

    pdb_col = next((c for c in pairs_df.columns if c.lower() in ["pdb", "pdb_id"]), None)
    uniprot_col = next(
        (c for c in pairs_df.columns if c.lower() in ["uniprot_id", "sp_primary", "uniprot"]),
        None,
    )

    if not pdb_col or not uniprot_col:
        raise ValueError(f"Columnas no identificadas. Disponibles: {list(pairs_df.columns)}")

    if max_pairs and max_pairs < len(pairs_df):
        pairs_df = pairs_df.head(max_pairs)
        print(f"Limitado a {max_pairs:,} pares")

    rows = [{
        "pdb_id": row[pdb_col],
        "uniprot_id": row[uniprot_col],
        "pdb_dir": str(Config.PDB_DIR),
        "af_dir": str(Config.AF_DIR),
    } for _, row in pairs_df.iterrows()]

    print(f"\nAnalizando {len(rows):,} pares...")

    results = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_analysis_worker,
        initargs=(sifts_index,)
    ) as executor:
        for result in tqdm(
            executor.map(analyze_structure_pair, rows, chunksize=50),
            total=len(rows),
            desc="Analizando",
            unit="pares",
        ):
            if result is not None:
                results.append(result)

    if not results:
        raise RuntimeError("No se pudieron analizar estructuras")

    results_df = pd.DataFrame(results)

    results_df.to_csv(output_dir / "structure_comparison_results.csv", index=False)

    print(f"\n{'=' * 60}")
    print(f"RESULTADOS")
    print(f"{'=' * 60}")
    print(f"Pares analizados: {len(results_df):,}")
    print(f"UniProt IDs unicos: {results_df['uniprot_id'].nunique()}")
    print(f"\nRMSD GLOBAL:")
    print(f"  Media: {results_df['global_rmsd'].mean():.2f} A")
    print(f"  Mediana: {results_df['global_rmsd'].median():.2f} A")
    print(f"  Std: {results_df['global_rmsd'].std():.2f} A")
    print(f"  < 1A: {(results_df['global_rmsd'] < 1).sum():,} ({(results_df['global_rmsd'] < 1).mean()*100:.1f}%)")
    print(f"  < 2A: {(results_df['global_rmsd'] < 2).sum():,} ({(results_df['global_rmsd'] < 2).mean()*100:.1f}%)")

    # Best/worst
    best = results_df.nsmallest(10, "global_rmsd")
    worst = results_df.nlargest(10, "global_rmsd")
    best.to_csv(output_dir / "best_predictions.csv", index=False)
    worst.to_csv(output_dir / "worst_predictions.csv", index=False)

    generate_statistics_report(results_df, output_dir)

    print(f"\nArchivos generados en: {output_dir}")
    return 0


def main():
    arg_parser = argparse.ArgumentParser(
        description="Analisis estadistico de estructuras PDB vs AlphaFold (SIFTS)"
    )
    arg_parser.add_argument("--max-pairs", type=int, default=None)
    arg_parser.add_argument("--workers", type=int, default=Config.DEFAULT_WORKERS)
    args = arg_parser.parse_args()

    return run_analysis(max_pairs=args.max_pairs, workers=args.workers)


if __name__ == "__main__":
    exit(main())
