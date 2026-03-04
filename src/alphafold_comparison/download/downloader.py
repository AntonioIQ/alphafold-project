#!/usr/bin/env python3
"""
Descarga masiva de estructuras proteicas de PDB y AlphaFold.

Lee pares desde id_list.csv (generado por rebuild_id_list.py) y descarga
los archivos PDB y AlphaFold que faltan.
"""

import argparse
import time

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

from alphafold_comparison.config import Config


def check_existing_files(pdb_dir: Path, af_dir: Path) -> tuple:
    """
    Verifica archivos existentes para evitar re-descargas.

    Returns:
        (set de PDB IDs existentes, set de UniProt IDs existentes)
    """
    existing_pdb = set()
    existing_af = set()

    if pdb_dir.exists():
        for file in pdb_dir.glob("*.ent"):
            pdb_id = file.stem.replace("pdb", "").upper()
            existing_pdb.add(pdb_id)

    if af_dir.exists():
        for file in af_dir.glob("*.pdb"):
            existing_af.add(file.stem)

    print(f"Archivos existentes: PDB={len(existing_pdb):,}, AlphaFold={len(existing_af):,}")
    return existing_pdb, existing_af


def download_file(url: str, output_path: Path, max_retries: int = 3) -> tuple:
    """Descarga un archivo con reintentos y validación."""
    if output_path.exists() and output_path.stat().st_size > 100:
        return True, "Ya existe"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60, stream=True)
            if response.status_code == 200:
                content = response.content
                if len(content) > 100:
                    with open(output_path, "wb") as f:
                        f.write(content)
                    return True, "Descargado"
                return False, "Archivo vacio"
            elif response.status_code == 404:
                return False, "No encontrado (404)"
            else:
                return False, f"HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False, "Timeout"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False, f"Error: {e}"

    return False, "Maximo de reintentos excedido"


def download_pdb(pdb_id: str, output_dir: Path) -> tuple:
    """Descarga estructura PDB desde RCSB."""
    output_path = output_dir / f"pdb{pdb_id.lower()}.ent"
    urls = [
        f"{Config.RCSB_BASE_URL}/{pdb_id}.pdb",
        f"{Config.RCSB_BASE_URL}/{pdb_id.lower()}.pdb",
        f"https://www.rcsb.org/pdb/files/{pdb_id}.pdb",
    ]
    for url in urls:
        success, message = download_file(url, output_path)
        if success:
            return True, message
    return False, "Todas las URLs fallaron"


def download_alphafold(uniprot_id: str, output_dir: Path) -> tuple:
    """Descarga modelo AlphaFold desde EBI."""
    output_path = output_dir / f"{uniprot_id}.pdb"
    for version in ["v6", "v5", "v4", "v3", "v2"]:
        url = f"{Config.ALPHAFOLD_BASE_URL}/AF-{uniprot_id}-F1-model_{version}.pdb"
        success, message = download_file(url, output_path)
        if success:
            return True, f"{message} ({version})"
    return False, "Todas las versiones fallaron"


def _download_single(args):
    """Descarga un par PDB+AF."""
    pdb_id, uniprot_id, pdb_dir, af_dir, pdb_exists, af_exists = args
    result = {
        "pdb_id": pdb_id,
        "uniprot_id": uniprot_id,
        "pdb_success": pdb_exists,
        "af_success": af_exists,
        "pdb_message": "Ya existia" if pdb_exists else "",
        "af_message": "Ya existia" if af_exists else "",
    }
    if not pdb_exists:
        success, message = download_pdb(pdb_id, pdb_dir)
        result["pdb_success"] = success
        result["pdb_message"] = message
    if not af_exists:
        success, message = download_alphafold(uniprot_id, af_dir)
        result["af_success"] = success
        result["af_message"] = message
    return result


def run_download(workers=None):
    """Ejecuta descarga desde id_list.csv."""
    workers = workers or Config.DOWNLOAD_WORKERS
    Config.ensure_dirs()

    print("=" * 60)
    print("PROTEIN STRUCTURE DOWNLOADER v3.0")
    print("=" * 60)

    if not Config.ID_LIST.exists():
        raise FileNotFoundError(
            f"No se encontro id_list.csv: {Config.ID_LIST}\n"
            "Ejecuta primero: python scripts/rebuild_id_list.py"
        )

    pairs_df = pd.read_csv(Config.ID_LIST)
    print(f"Pares en id_list: {len(pairs_df):,}")
    print(f"UniProt unicos: {pairs_df['SP_PRIMARY'].nunique():,}")

    existing_pdb, existing_af = check_existing_files(Config.PDB_DIR, Config.AF_DIR)

    # Preparar lista de descargas
    download_args = []
    already_complete = 0
    for _, row in pairs_df.iterrows():
        pdb_id = row["PDB"]
        uniprot_id = row["SP_PRIMARY"]
        pdb_exists = pdb_id in existing_pdb
        af_exists = uniprot_id in existing_af

        if pdb_exists and af_exists:
            already_complete += 1
            continue

        download_args.append((
            pdb_id, uniprot_id, Config.PDB_DIR, Config.AF_DIR,
            pdb_exists, af_exists
        ))

    print(f"Ya completos: {already_complete:,}")
    print(f"Pendientes de descarga: {len(download_args):,}")

    if not download_args:
        print("Todas las estructuras ya descargadas.")
        return 0

    print(f"\nDescargando con {workers} workers...")
    start_time = time.time()

    all_results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in tqdm(
            executor.map(_download_single, download_args),
            total=len(download_args),
            desc="Descargando",
            unit="pares",
        ):
            all_results.append(result)

    elapsed = time.time() - start_time

    # Estadísticas
    pdb_ok = sum(1 for r in all_results if r["pdb_success"])
    af_ok = sum(1 for r in all_results if r["af_success"])
    both_ok = sum(1 for r in all_results if r["pdb_success"] and r["af_success"])

    print(f"\n{'=' * 60}")
    print(f"DESCARGA COMPLETADA en {elapsed / 60:.1f} minutos")
    print(f"{'=' * 60}")
    print(f"Descargados: {len(all_results):,}")
    print(f"PDB obtenidos: {pdb_ok:,}/{len(all_results):,}")
    print(f"AlphaFold obtenidos: {af_ok:,}/{len(all_results):,}")
    print(f"Pares completos: {both_ok:,}")
    print(f"Total disponibles: {already_complete + both_ok:,}/{len(pairs_df):,}")

    # Guardar reporte
    results_df = pd.DataFrame(all_results)
    report_path = Config.RAW_DIR / "download_report_detailed.csv"
    results_df.to_csv(report_path, index=False)
    print(f"Reporte: {report_path}")

    # Actualizar id_list: solo pares donde ambos archivos existen
    existing_pdb_new, existing_af_new = check_existing_files(Config.PDB_DIR, Config.AF_DIR)
    valid_pairs = pairs_df[
        pairs_df['PDB'].isin(existing_pdb_new) &
        pairs_df['SP_PRIMARY'].isin(existing_af_new)
    ]
    valid_pairs.to_csv(Config.ID_LIST, index=False)
    print(f"id_list.csv actualizado: {len(valid_pairs):,} pares con ambos archivos")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Descarga estructuras PDB y AlphaFold")
    parser.add_argument("--workers", type=int, default=Config.DOWNLOAD_WORKERS)
    args = parser.parse_args()

    return run_download(workers=args.workers)


if __name__ == "__main__":
    exit(main())
