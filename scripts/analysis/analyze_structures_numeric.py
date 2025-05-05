#!/usr/bin/env python3
"""
Script para análisis estadístico de estructuras PDB vs AlphaFold (solo numérico)
"""

import os
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, Superimposer
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings

# Ignorar advertencias de Bio.PDB
warnings.filterwarnings("ignore")

# Configuración
DATA_DIR = "/media/antonio-tapia/jatg/protein_structures"
PDB_DIR = os.path.join(DATA_DIR, "pdb_files")
AF_DIR = os.path.join(DATA_DIR, "alphafold_models")
OUTPUT_DIR = os.path.join(DATA_DIR, "analysis_results")
PAIRS_FILE = os.path.join(DATA_DIR, "id_list.csv")  # Cambiado a id_list.csv
MAX_PAIRS = 1000  # Limitado a 1000 pares para prueba inicial

# Crear directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inicializar parser
parser = PDBParser(QUIET=True)

def analyze_structure_pair(row):
    """Analiza un par de estructuras PDB y AlphaFold"""
    # Determinar nombres de columnas
    pdb_col = next((col for col in row.keys() if 'pdb' in col.lower()), None)
    uniprot_col = next((col for col in row.keys() if 'uniprot' in col.lower() or 'sp_primary' in col.lower()), None)
    
    if not pdb_col or not uniprot_col:
        return None
    
    pdb_id = row[pdb_col]
    uniprot_id = row[uniprot_col]
    
    # Construir rutas a los archivos
    pdb_file = os.path.join(PDB_DIR, f"pdb{pdb_id.lower()}.ent")
    af_file = os.path.join(AF_DIR, f"{uniprot_id}.pdb")
    
    # Verificar que ambos archivos existen
    if not (os.path.exists(pdb_file) and os.path.exists(af_file)):
        return None
    
    try:
        # Cargar estructuras
        pdb_structure = parser.get_structure("pdb", pdb_file)
        af_structure = parser.get_structure("af", af_file)
        
        # Por simplicidad, usamos solo la primera cadena
        chain_ids = [chain.id for chain in pdb_structure[0]]
        if not chain_ids:
            return None
        
        chain_id = chain_ids[0]
        
        # Verificar si la cadena existe en ambas estructuras
        if chain_id not in [c.id for c in af_structure[0]]:
            # Intentar con cadena A como alternativa
            if 'A' in [c.id for c in af_structure[0]]:
                chain_id = 'A'
            else:
                return None
        
        # Extraer CAs
        pdb_cas = []
        pdb_res_ids = []
        for res in pdb_structure[0][chain_id]:
            if "CA" in res:
                pdb_cas.append(res["CA"])
                pdb_res_ids.append(res.id[1])
        
        af_cas = []
        af_res_ids = []
        for res in af_structure[0][chain_id]:
            if "CA" in res:
                af_cas.append(res["CA"])
                af_res_ids.append(res.id[1])
        
        # Encontrar residuos comunes
        common_res_ids = sorted(set(pdb_res_ids) & set(af_res_ids))
        
        if len(common_res_ids) < 10:  # Requerir al menos 10 residuos comunes
            return None
        
        # Filtrar CAs para residuos comunes
        pdb_cas_filtered = []
        af_cas_filtered = []
        
        for res_id in common_res_ids:
            pdb_idx = pdb_res_ids.index(res_id)
            af_idx = af_res_ids.index(res_id)
            pdb_cas_filtered.append(pdb_cas[pdb_idx])
            af_cas_filtered.append(af_cas[af_idx])
        
        # Alinear estructuras
        super_imposer = Superimposer()
        super_imposer.set_atoms(pdb_cas_filtered, af_cas_filtered)
        super_imposer.apply(af_cas_filtered)
        
        # Calcular RMSD por residuo
        rmsd_per_residue = []
        for pdb_ca, af_ca in zip(pdb_cas_filtered, af_cas_filtered):
            dist = np.linalg.norm(pdb_ca.get_coord() - af_ca.get_coord())
            rmsd_per_residue.append(dist)
        
        # Calcular estadísticas
        global_rmsd = super_imposer.rms
        mean_rmsd = np.mean(rmsd_per_residue)
        std_rmsd = np.std(rmsd_per_residue)
        max_rmsd = np.max(rmsd_per_residue)
        high_rmsd_count = np.sum(np.array(rmsd_per_residue) > 2.0)
        percent_high_rmsd = (high_rmsd_count / len(rmsd_per_residue)) * 100
        
        # Extraer información de estructura
        protein_length = len(common_res_ids)
        
        # Diccionario para resultados
        return {
            'pdb_id': pdb_id,
            'uniprot_id': uniprot_id,
            'chain_id': chain_id,
            'protein_length': protein_length,
            'global_rmsd': global_rmsd,
            'mean_rmsd': mean_rmsd,
            'std_rmsd': std_rmsd,
            'max_rmsd': max_rmsd,
            'high_rmsd_count': high_rmsd_count,
            'percent_high_rmsd': percent_high_rmsd
        }
    
    except Exception as e:
        print(f"Error procesando {pdb_id} - {uniprot_id}: {str(e)}")
        return None

def main():
    # Cargar lista de pares
    if not os.path.exists(PAIRS_FILE):
        print(f"No se encontró el archivo de pares: {PAIRS_FILE}")
        return
    
    pairs_df = pd.read_csv(PAIRS_FILE)
    print(f"Se encontraron {len(pairs_df)} pares en {PAIRS_FILE}")
    
    # Verificar que tenemos las columnas necesarias
    columns = pairs_df.columns
    pdb_col = next((c for c in columns if 'pdb' in c.lower()), None)
    uniprot_col = next((c for c in columns if 'uniprot' in c.lower() or 'sp_primary' in c.lower() or 'sp' in c.lower()), None)
    
    if not pdb_col or not uniprot_col:
        print(f"No se pudieron identificar las columnas necesarias en {PAIRS_FILE}")
        print(f"Columnas disponibles: {columns.tolist()}")
        return
    
    print(f"Usando columna '{pdb_col}' para IDs PDB")
    print(f"Usando columna '{uniprot_col}' para IDs UniProt")
    
    # Limitar para análisis inicial
    if MAX_PAIRS and MAX_PAIRS < len(pairs_df):
        pairs_df = pairs_df.head(MAX_PAIRS)
        print(f"Análisis limitado a {MAX_PAIRS} pares")
    
    # Procesar pares en paralelo
    print("Analizando pares de estructuras...")
    results = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(analyze_structure_pair, row) for row in pairs_df.to_dict('records')]
        
        for future in tqdm(futures, total=len(pairs_df)):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error en procesamiento: {str(e)}")
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(results)
    
    # Guardar resultados completos
    csv_output = os.path.join(OUTPUT_DIR, "structure_comparison_results.csv")
    results_df.to_csv(csv_output, index=False)
    print(f"Resultados guardados en {csv_output}")
    
    # Estadísticas básicas
    print("\n=== Estadísticas básicas ===")
    print(f"Pares analizados con éxito: {len(results_df)}")
    
    # Métricas clave
    print("\nRMSD Global:")
    rmsd_stats = results_df['global_rmsd'].describe()
    for stat, value in rmsd_stats.items():
        print(f"  {stat}: {value:.4f}")
    
    print("\nRMSD Medio por residuo:")
    mean_rmsd_stats = results_df['mean_rmsd'].describe()
    for stat, value in mean_rmsd_stats.items():
        print(f"  {stat}: {value:.4f}")
    
    print("\nPorcentaje de residuos con RMSD > 2Å:")
    high_rmsd_stats = results_df['percent_high_rmsd'].describe()
    for stat, value in high_rmsd_stats.items():
        print(f"  {stat}: {value:.4f}%")
    
    # Identificar las 10 mejores y 10 peores predicciones
    best_predictions = results_df.nsmallest(10, 'global_rmsd')
    worst_predictions = results_df.nlargest(10, 'global_rmsd')
    
    print("\n=== Top 10 Mejores Predicciones ===")
    print(best_predictions[['pdb_id', 'uniprot_id', 'global_rmsd', 'protein_length']])
    
    print("\n=== Top 10 Peores Predicciones ===")
    print(worst_predictions[['pdb_id', 'uniprot_id', 'global_rmsd', 'protein_length']])
    
    # Guardar estos subconjuntos
    best_predictions.to_csv(os.path.join(OUTPUT_DIR, "best_predictions.csv"), index=False)
    worst_predictions.to_csv(os.path.join(OUTPUT_DIR, "worst_predictions.csv"), index=False)
    
    # Guardar estadísticas en un archivo
    with open(os.path.join(OUTPUT_DIR, "summary_statistics.txt"), 'w') as f:
        f.write("=== Análisis Estadístico de Estructuras PDB vs AlphaFold ===\n\n")
        f.write(f"Total de pares analizados: {len(results_df)}\n\n")
        
        f.write("Estadísticas de RMSD Global:\n")
        for stat, value in rmsd_stats.items():
            f.write(f"  {stat}: {value:.4f}\n")
        
        f.write("\nEstadísticas de RMSD Medio por Residuo:\n")
        for stat, value in mean_rmsd_stats.items():
            f.write(f"  {stat}: {value:.4f}\n")
        
        f.write("\nEstadísticas de Porcentaje de Residuos con RMSD > 2Å:\n")
        for stat, value in high_rmsd_stats.items():
            f.write(f"  {stat}: {value:.4f}%\n")

if __name__ == "__main__":
    main()