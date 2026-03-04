#!/usr/bin/env python3
"""
Descarga modelos AlphaFold adicionales hasta alcanzar el target.

Estrategia: prueba UniProt IDs del SIFTS que aún no tenemos,
con timeout corto para saltar 404s rápidamente.
"""

import sys
import time
import random
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

AF_DIR = Path("data/raw/alphafold_models")
SIFTS_FILE = Path("data/raw/mappings/pdb_chain_uniprot.csv")
TARGET = 10500  # un poco más del target para compensar pérdidas
WORKERS = 20
BATCH_SIZE = 5000


def download_af(uniprot_id: str) -> tuple:
    """Intenta descargar AF model. Retorna (uniprot_id, success)."""
    output = AF_DIR / f"{uniprot_id}.pdb"
    if output.exists() and output.stat().st_size > 100:
        return uniprot_id, True

    for version in ["v6", "v5", "v4"]:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_{version}.pdb"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 100:
                with open(output, "wb") as f:
                    f.write(r.content)
                return uniprot_id, True
            elif r.status_code == 404:
                if version == "v4":
                    continue  # try older versions
                return uniprot_id, False
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            continue
    return uniprot_id, False


def main():
    AF_DIR.mkdir(parents=True, exist_ok=True)

    # Existing AF models
    existing = {f.stem for f in AF_DIR.glob("*.pdb")}
    print(f"AF models existentes: {len(existing)}")

    if len(existing) >= TARGET:
        print(f"Ya tenemos >= {TARGET} modelos. Nada que hacer.")
        return

    # Load SIFTS UniProt IDs
    sifts = pd.read_csv(SIFTS_FILE, comment="#", low_memory=False)
    sifts.columns = sifts.columns.str.strip()
    all_uniprots = list(set(sifts["SP_PRIMARY"].dropna().unique()) - existing)
    random.shuffle(all_uniprots)  # randomizar para diversidad

    print(f"Candidatos a probar: {len(all_uniprots)}")
    needed = TARGET - len(existing)
    print(f"Necesitamos: ~{needed} más")

    # Descargar en lotes
    total_new = 0
    total_tried = 0

    for batch_start in range(0, len(all_uniprots), BATCH_SIZE):
        if total_new >= needed:
            break

        batch = all_uniprots[batch_start : batch_start + BATCH_SIZE]
        print(f"\n--- Lote {batch_start//BATCH_SIZE + 1}: probando {len(batch)} UniProt IDs ---")

        batch_ok = 0
        batch_fail = 0

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(download_af, uid): uid for uid in batch}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Descargando", unit="prot"):
                uid, success = future.result()
                if success:
                    batch_ok += 1
                else:
                    batch_fail += 1

        total_new += batch_ok
        total_tried += len(batch)
        current_total = len(existing) + total_new

        print(f"  Lote: {batch_ok} OK, {batch_fail} fail ({batch_ok/(batch_ok+batch_fail)*100:.1f}% éxito)")
        print(f"  Total acumulado: {current_total} AF models")
        print(f"  Progreso: {current_total}/{TARGET} ({current_total/TARGET*100:.1f}%)")

        if total_new >= needed:
            break

        # Si la tasa de éxito es muy baja, necesitaremos más lotes
        success_rate = batch_ok / len(batch) if batch else 0
        if success_rate < 0.01:
            print(f"  ADVERTENCIA: tasa de éxito muy baja ({success_rate*100:.2f}%). Continuando...")

    # Resumen final
    final_count = len(list(AF_DIR.glob("*.pdb")))
    print(f"\n{'='*60}")
    print(f"DESCARGA COMPLETADA")
    print(f"{'='*60}")
    print(f"Probados: {total_tried}")
    print(f"Nuevos descargados: {total_new}")
    print(f"Total AF models: {final_count}")
    print(f"Target: {TARGET}")
    print(f"{'ALCANZADO' if final_count >= TARGET else 'NO ALCANZADO'}")


if __name__ == "__main__":
    main()
