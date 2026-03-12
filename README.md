# AlphaFold vs PDB: Análisis Comparativo de Estructuras Proteicas

Evaluación sistemática de la precisión de AlphaFold2 comparando estructuras proteicas experimentales (PDB) contra predicciones computacionales, utilizando **mapeo SIFTS** para la correspondencia correcta de residuos.

## Hallazgo Principal

Con mapeo SIFTS correcto (`PDB_BEG → SP_BEG`), AlphaFold2 demuestra **precisión comparable a la experimental**:

| Métrica | Con mapeo SIFTS | Sin mapeo (erróneo) |
|---------|----------------|---------------------|
| RMSD medio | **1.05 Å** | 29.94 Å |
| RMSD mediana | **0.64 Å** | 22.54 Å |
| < 1 Å | **70.2%** | 9.3% |
| < 2 Å | **87.7%** | 13.9% |
| > 10 Å | **0.4%** | 80.9% |

**No existe un "sweet spot" de tamaño**: la precisión es alta y uniforme en todos los rangos. El aparente rango óptimo de 250-300 residuos reportado previamente era un artefacto del desfase de numeración PDB/UniProt.

## Por qué SIFTS es esencial

Los archivos PDB usan numeraciones de residuos que frecuentemente **no coinciden** con UniProt/AlphaFold:

- **Péptidos señal**: La lisozima (P00698) se numera 1-129 en PDB, pero 19-147 en UniProt (18 residuos de péptido señal)
- **Cadenas múltiples**: PDBs multiméricos contienen varias proteínas; sin SIFTS, se comparan cadenas incorrectas
- **Resultado**: Sin mapeo, el RMSD de ~0.44 Å se infla artificialmente a ~18 Å

## Estructura del Proyecto

```
alphafold-project/
├── src/alphafold_comparison/     # Paquete Python instalable
│   ├── config.py                 # Configuración centralizada (.env)
│   ├── download/                 # Descarga PDB + AlphaFold
│   ├── preprocessing/            # Mapeo SIFTS + superposición Kabsch
│   ├── analysis/                 # Análisis estructural y atómico
│   └── visualization/            # Funciones de plotting
├── scripts/
│   └── rebuild_id_list.py        # Genera id_list diverso desde SIFTS
├── notebooks/                    # Narrativa científica
├── reports/latex/                # Reporte LaTeX completo
├── data/                         # Datos (~27 GB, gitignored)
│   ├── raw/mappings/             # SIFTS + id_list.csv
│   ├── raw/pdb_files/            # Estructuras experimentales
│   ├── raw/alphafold_models/     # Predicciones AlphaFold2
│   ├── processed/                # Índice de calidad certificado
│   └── results/                  # Resultados consolidados
├── .env.example                  # Template de configuración
├── pyproject.toml                # Dependencias y metadata
├── Makefile                      # Automatización
└── anterior/                     # Versión previa (gitignored)
```

## Instalación (Debian 12)

### Dependencias del sistema

```bash
sudo apt-get update
sudo apt-get install -y python3-dev python3-venv python3-pip \
    texlive-full texlive-lang-spanish latexmk git make
```

### Setup del proyecto

```bash
git clone <repo-url>
cd alphafold-project

# Configurar entorno
cp .env.example .env
# Editar .env con tus rutas

# Instalar (crea venv + instala paquete)
make setup
```

## Uso

### Pipeline completo

```bash
make all          # download → process → analyze
```

### Por fases

```bash
make download     # Descargar estructuras PDB y AlphaFold
make process      # Mapeo SIFTS + superposición Kabsch
make analyze      # Análisis estadístico
```

### CLI directa

```bash
python -m alphafold_comparison download --workers 15
python -m alphafold_comparison process --workers 8
python -m alphafold_comparison analyze
python -m alphafold_comparison config   # Ver configuración
```

### Notebooks

```bash
jupyter lab notebooks/
```

### Compilar reporte LaTeX

```bash
make report       # Genera PDF en reports/latex/main.pdf
```

## Pipeline

```
SIFTS (pdb_chain_uniprot.csv)
       │
       ▼
┌─────────────────────────────┐
│  DIVERSIFICACIÓN            │
│  10,432 UniProt IDs únicos  │
│  1 PDB representativo c/u   │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  DESCARGA                   │
│  PDB + AlphaFold models     │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  PROCESAMIENTO (SIFTS)      │
│  1. Mapeo cadena correcta   │
│  2. Mapeo PDB→UniProt pos   │
│  3. Extracción CAs mapeados │
│  4. Superposición Kabsch     │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  ANÁLISIS                   │
│  RMSD por tamaño, CHONSP,   │
│  backbone vs sidechain      │
└─────────────────────────────┘
```

## Dependencias

- Python >= 3.10
- NumPy, Pandas, SciPy, BioPython
- Matplotlib, Seaborn
- Requests, tqdm, python-dotenv
- LaTeX (para reporte): texlive-full

Ver [pyproject.toml](pyproject.toml) para versiones específicas.

## Autor

Antonio Tapia — Maestría en Ciencia de Datos, ITAM
