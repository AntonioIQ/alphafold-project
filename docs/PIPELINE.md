# Pipeline Técnico: AlphaFold Comparison

## Fase 1: Diversificación del dataset (`scripts/rebuild_id_list.py`)

**Input**: Archivo SIFTS (`data/raw/mappings/pdb_chain_uniprot.csv`)
**Output**: `data/raw/mappings/id_list.csv`

- Lee SIFTS completo (~870K entradas, 68K UniProt IDs únicos)
- Selecciona 1 PDB representativo por UniProt ID (máxima cobertura)
- Prioriza pares que ya tienen modelos AlphaFold descargados
- Incluye columna CHAIN desde SIFTS
- Target: 10,000 proteínas con UniProt IDs únicos

## Fase 2: Descarga (`src/alphafold_comparison/download/`)

**Input**: `id_list.csv` (columnas: PDB, SP_PRIMARY, CHAIN)
**Output**: `data/raw/pdb_files/`, `data/raw/alphafold_models/`

- PDB descargado de RCSB (3 URLs de fallback)
- AlphaFold descargado de EBI (versiones v4, v3, v2)
- Multi-threaded (15 workers por defecto)
- Solo descarga archivos que no existen ya
- Tras descarga, actualiza id_list.csv eliminando pares sin archivos

## Fase 3: Procesamiento con SIFTS (`src/alphafold_comparison/preprocessing/processor.py`)

**Input**: `id_list.csv` + archivos PDB/AlphaFold + SIFTS
**Output**: `data/processed/quality_structures_index.csv`

### Paso 1: Mapeo SIFTS
- Construye índice: `(PDB_upper, SP) → [(chain, pdb_beg, pdb_end, sp_beg, sp_end)]`
- Para cada par, obtiene cadena correcta + mapeo `{pdb_pos: uniprot_pos}`
- Esto resuelve: numeración diferente (ej: PDB 1-129 → UniProt 19-147) y selección de cadena en PDBs multiméricos

### Paso 2: Extracción de CAs
- PDB: solo cadena SIFTS, posiciones mapeadas a numeración UniProt
- AlphaFold: cadena A, numeración UniProt directa
- Posiciones comunes = intersección de posiciones UniProt

### Paso 3: Superposición Kabsch
- BioPython Superimposer sobre CAs de posiciones comunes
- Cálculo de RMSD global
- Multiprocessing con ProcessPoolExecutor

### Resultado esperado
- RMSD medio ~1 Å, mediana ~0.64 Å
- 87.7% < 2 Å, 70.2% < 1 Å

## Fase 4: Análisis (`src/alphafold_comparison/analysis/`)

### Análisis Estructural (`structural.py`)
- RMSD global y por residuo (CA atoms) con mapeo SIFTS
- Estadísticas: media, mediana, std, percentiles
- Categorización por tamaño de proteína
- No hay "sweet spot": precisión uniforme por tamaño

### Análisis Atómico (`atomic.py`)
- RMSD por elemento químico (CHONSP) con mapeo SIFTS
- Backbone (N, CA, C, O) vs sidechain
- Cadena SIFTS correcta + posiciones mapeadas

## Ejecución

```bash
# Pipeline completo
make all                    # download → process → analyze

# Por fases
python scripts/rebuild_id_list.py          # Regenerar id_list diverso
python -m alphafold_comparison download    # Descargar estructuras
python -m alphafold_comparison process     # Procesar con SIFTS
python -m alphafold_comparison analyze     # Análisis estadístico

# Reporte
make report                 # Compilar LaTeX
```
