#!/usr/bin/env python3
"""
Script simplificado para descargar estructuras proteicas
"""

import os
import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# CONFIGURACIÓN - CAMBIAR ESTAS RUTAS SEGÚN TU SISTEMA
OUTPUT_DIR = "/media/antonio-tapia/jatg/protein_structures"  # Ruta a tu disco duro externo
SIFTS_FILE = "/home/antonio-tapia/pdb_chain_uniprot.csv"  # Archivo SIFTS que descargaste
MAX_STRUCTURES = 10000  # Número de estructuras a descargar
MAX_WORKERS = 20  # Descargas paralelas

# Crear directorios de salida
pdb_dir = os.path.join(OUTPUT_DIR, "pdb_files")
af_dir = os.path.join(OUTPUT_DIR, "alphafold_models")
os.makedirs(pdb_dir, exist_ok=True)
os.makedirs(af_dir, exist_ok=True)

def download_file(url, output_path):
    """Descarga un archivo desde una URL"""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        print(f"Error descargando {url}: {str(e)}")
        return False

def download_pdb(pdb_id):
    """Descarga una estructura del PDB"""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = os.path.join(pdb_dir, f"pdb{pdb_id.lower()}.ent")
    return download_file(url, output_path)

def download_alphafold(uniprot_id):
    """Descarga un modelo de AlphaFold"""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    output_path = os.path.join(af_dir, f"{uniprot_id}.pdb")
    
    if download_file(url, output_path):
        return True
    
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v3.pdb"
    return download_file(url, output_path)

def main():
    print(f"Procesando archivo SIFTS: {SIFTS_FILE}")
    
    # Leer el archivo CSV ignorando la primera línea
    df = pd.read_csv(SIFTS_FILE, skiprows=1)

    print(f"Datos cargados, forma: {df.shape}")
    
    # Verificar si tenemos las columnas necesarias
    if 'PDB' not in df.columns or 'SP_PRIMARY' not in df.columns:
        print("No se encontraron las columnas necesarias (PDB, SP_PRIMARY)")
        print("Columnas encontradas:", df.columns)
        return
    
    # Extraer pares PDB-UniProt y eliminar duplicados
    pairs_df = df[['PDB', 'SP_PRIMARY']].drop_duplicates().dropna()
    
    # Limitar cantidad
    if len(pairs_df) > MAX_STRUCTURES:
        pairs_df = pairs_df.head(MAX_STRUCTURES)
    
    # Guardar la lista para referencia futura
    id_list_path = os.path.join(OUTPUT_DIR, "id_list.csv")
    pairs_df.to_csv(id_list_path, index=False)
    
    print(f"Lista de {len(pairs_df)} pares PDB-UniProt guardada en {id_list_path}")
    
    # Preparar lista de pares para descarga
    pdb_ids = pairs_df['PDB'].tolist()
    uniprot_ids = pairs_df['SP_PRIMARY'].tolist()
    
    # Descargar estructuras PDB
    print("\nDescargando estructuras PDB...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pdb_results = list(tqdm(
            executor.map(download_pdb, pdb_ids),
            total=len(pdb_ids),
            desc="Descargando PDB"
        ))
    
    # Descargar modelos AlphaFold
    print("\nDescargando modelos AlphaFold...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        af_results = list(tqdm(
            executor.map(download_alphafold, uniprot_ids),
            total=len(uniprot_ids),
            desc="Descargando AlphaFold"
        ))
    
    # Generar reporte
    pdb_success = sum(pdb_results)
    af_success = sum(af_results)
    both_success = sum(1 for p, a in zip(pdb_results, af_results) if p and a)
    
    print("\n=== Resumen de descarga ===")
    print(f"Total de pares: {len(pdb_ids)}")
    print(f"Estructuras PDB descargadas: {pdb_success}")
    print(f"Modelos AlphaFold descargados: {af_success}")
    print(f"Pares completos (ambos descargados): {both_success}")
    
    # Guardar reporte
    report_path = os.path.join(OUTPUT_DIR, "download_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de pares: {len(pdb_ids)}\n")
        f.write(f"Estructuras PDB descargadas: {pdb_success}\n")
        f.write(f"Modelos AlphaFold descargados: {af_success}\n")
        f.write(f"Pares completos (ambos descargados): {both_success}\n")
    
    print(f"\nReporte guardado en {report_path}")

if __name__ == "__main__":
    main()
