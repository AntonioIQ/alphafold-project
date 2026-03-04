#!/usr/bin/env python3
"""
Regenera id_list.csv con máxima diversidad de UniProt IDs.

Estrategia:
1. Lee SIFTS completo
2. Para cada UniProt ID, selecciona EL PDB con mayor cobertura de residuos
3. Incluye columna CHAIN desde SIFTS
4. Target: 10,000+ UniProt IDs diversos
5. Prioriza UniProt IDs que ya tienen modelo AlphaFold descargado
"""
import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from alphafold_comparison.config import Config


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

    print("=" * 60)
    print("REBUILD ID_LIST - Diversificacion de dataset")
    print("=" * 60)

    # 1. Cargar SIFTS
    print(f"\nCargando SIFTS: {Config.SIFTS_FILE}")
    sifts = pd.read_csv(str(Config.SIFTS_FILE), comment='#', low_memory=False)
    sifts.columns = sifts.columns.str.strip()
    print(f"  {len(sifts):,} filas")

    # Limpiar
    sifts['PDB'] = sifts['PDB'].astype(str).str.upper().str.strip()
    sifts['SP_PRIMARY'] = sifts['SP_PRIMARY'].astype(str).str.strip()
    sifts['CHAIN'] = sifts['CHAIN'].astype(str).str.strip()

    # Filtrar IDs válidos
    sifts = sifts[sifts['PDB'].str.match(r'^[0-9A-Z]{4}$')]
    sifts = sifts[sifts['SP_PRIMARY'].str.len() >= 6]
    print(f"  {sifts['SP_PRIMARY'].nunique():,} UniProt IDs unicos")
    print(f"  {sifts['PDB'].nunique():,} PDB IDs unicos")

    # 2. Calcular cobertura: residuos mapeados por (PDB, SP_PRIMARY, CHAIN)
    try:
        sifts['coverage'] = (
            sifts['RES_END'].astype(int) - sifts['RES_BEG'].astype(int) + 1
        )
    except (ValueError, TypeError):
        sifts['coverage'] = 1

    # Agregar cobertura por (PDB, SP_PRIMARY, CHAIN)
    coverage = sifts.groupby(['PDB', 'SP_PRIMARY', 'CHAIN'])['coverage'].sum().reset_index()
    coverage.columns = ['PDB', 'SP_PRIMARY', 'CHAIN', 'total_coverage']

    # 3. Para cada UniProt, seleccionar el PDB con mayor cobertura
    best_pdb = coverage.sort_values('total_coverage', ascending=False).drop_duplicates(
        subset=['SP_PRIMARY'], keep='first'
    )
    print(f"\n  Mejor PDB por UniProt: {len(best_pdb):,} pares")

    # 4. Verificar qué AlphaFold models ya están descargados
    af_dir = Config.AF_DIR
    existing_af = set()
    if af_dir.exists():
        for f in af_dir.glob("*.pdb"):
            existing_af.add(f.stem)
    print(f"  AF models existentes: {len(existing_af):,}")

    # Verificar qué PDB files ya están descargados
    pdb_dir = Config.PDB_DIR
    existing_pdb = set()
    if pdb_dir.exists():
        for f in pdb_dir.glob("*.ent"):
            existing_pdb.add(f.stem.replace("pdb", "").upper())
    print(f"  PDB files existentes: {len(existing_pdb):,}")

    # 5. Priorizar: ya tiene AF > ya tiene PDB > ninguno
    best_pdb['has_af'] = best_pdb['SP_PRIMARY'].isin(existing_af)
    best_pdb['has_pdb'] = best_pdb['PDB'].isin(existing_pdb)
    best_pdb['priority'] = (
        best_pdb['has_af'].astype(int) * 2 +
        best_pdb['has_pdb'].astype(int) +
        best_pdb['total_coverage'] / best_pdb['total_coverage'].max()
    )
    best_pdb = best_pdb.sort_values('priority', ascending=False)

    # 6. Seleccionar top N
    selected = best_pdb.head(target).copy()

    # Stats
    has_both = ((selected['has_af']) & (selected['has_pdb'])).sum()
    has_af_only = ((selected['has_af']) & (~selected['has_pdb'])).sum()
    has_pdb_only = ((~selected['has_af']) & (selected['has_pdb'])).sum()
    has_neither = ((~selected['has_af']) & (~selected['has_pdb'])).sum()

    print(f"\n=== SELECCION: {len(selected):,} pares ===")
    print(f"  Ya tienen PDB + AF: {has_both:,}")
    print(f"  Solo AF: {has_af_only:,}")
    print(f"  Solo PDB: {has_pdb_only:,}")
    print(f"  Nuevas descargas: {has_neither:,}")
    print(f"  Descargas AF necesarias: {has_af_only + has_neither + has_pdb_only:,}")
    print(f"  Descargas PDB necesarias: {has_pdb_only + has_neither + has_af_only:,}")

    need_af = (~selected['has_af']).sum()
    need_pdb = (~selected['has_pdb']).sum()
    print(f"\n  TOTAL descargas necesarias:")
    print(f"    AF models nuevos: {need_af:,}")
    print(f"    PDB files nuevos: {need_pdb:,}")

    # Distribución de cobertura
    print(f"\n  Cobertura de residuos:")
    print(f"    Media: {selected['total_coverage'].mean():.0f}")
    print(f"    Mediana: {selected['total_coverage'].median():.0f}")
    print(f"    Min: {selected['total_coverage'].min():.0f}")
    print(f"    Max: {selected['total_coverage'].max():.0f}")

    # 7. Guardar id_list.csv
    output = selected[['PDB', 'SP_PRIMARY', 'CHAIN']].copy()
    output_path = Config.ID_LIST

    # Backup del anterior
    if output_path.exists():
        backup = output_path.with_suffix('.csv.bak')
        os.rename(output_path, backup)
        print(f"\n  Backup: {backup}")

    output.to_csv(output_path, index=False)
    print(f"  Nuevo id_list.csv: {output_path}")
    print(f"  {len(output):,} pares, {output['SP_PRIMARY'].nunique():,} UniProt IDs unicos")


if __name__ == '__main__':
    main()
