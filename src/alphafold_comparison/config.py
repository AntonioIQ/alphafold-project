"""
Configuración centralizada del proyecto.

Carga variables de entorno desde .env y construye todos los paths
dinámicamente. Ningún otro módulo debe tener paths hardcodeados.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Encontrar la raíz del proyecto (donde está pyproject.toml)
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


class Config:
    """Configuración centralizada del proyecto AlphaFold Comparison."""

    # --- Rutas base ---
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(_project_root)))
    DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))

    # --- Datos crudos ---
    RAW_DIR = DATA_DIR / "raw"
    PDB_DIR = RAW_DIR / "pdb_files"
    AF_DIR = RAW_DIR / "alphafold_models"
    MAPPINGS_DIR = RAW_DIR / "mappings"
    SIFTS_FILE = Path(os.getenv("SIFTS_FILE", str(MAPPINGS_DIR / "pdb_chain_uniprot.csv")))
    ID_LIST = MAPPINGS_DIR / "id_list.csv"

    # --- Datos procesados ---
    PROCESSED_DIR = DATA_DIR / "processed"
    CLEAN_PDB_DIR = PROCESSED_DIR / "clean_pdb"
    CLEAN_AF_DIR = PROCESSED_DIR / "clean_alphafold"
    ALIGNED_DIR = PROCESSED_DIR / "aligned_pairs"
    REPORTS_DIR = PROCESSED_DIR / "reports"
    QUALITY_INDEX = PROCESSED_DIR / "quality_structures_index.csv"

    # --- Datos validados ---
    VALIDATED_DIR = DATA_DIR / "validated"
    VALIDATED_INDEX = VALIDATED_DIR / "validated_pairs_index.csv"

    # --- Resultados ---
    RESULTS_DIR = DATA_DIR / "results"

    # --- Parámetros de descarga ---
    DOWNLOAD_TARGET = int(os.getenv("DOWNLOAD_TARGET", "50000"))
    DOWNLOAD_WORKERS = int(os.getenv("DOWNLOAD_WORKERS", "15"))
    DOWNLOAD_CHUNK_SIZE = int(os.getenv("DOWNLOAD_CHUNK_SIZE", "1000"))

    # --- Parámetros de procesamiento ---
    DEFAULT_WORKERS = int(os.getenv("DEFAULT_WORKERS", "8"))
    MIN_RESIDUES = int(os.getenv("MIN_RESIDUES", "3"))
    MIN_ATOMS_PER_RES = int(os.getenv("MIN_ATOMS_PER_RES", "1"))
    MIN_ALIGNMENT_ATOMS = int(os.getenv("MIN_ALIGNMENT_ATOMS", "3"))
    MIN_COMMON_RESIDUES = int(os.getenv("MIN_COMMON_RESIDUES", "10"))

    # --- Constantes estructurales ---
    BACKBONE_ATOMS = {"N", "CA", "C", "O", "CB"}

    # --- URLs de APIs ---
    RCSB_BASE_URL = "https://files.rcsb.org/download"
    ALPHAFOLD_BASE_URL = "https://alphafold.ebi.ac.uk/files"

    @classmethod
    def ensure_dirs(cls):
        """Crear todos los directorios necesarios."""
        for d in [
            cls.PDB_DIR, cls.AF_DIR, cls.MAPPINGS_DIR,
            cls.CLEAN_PDB_DIR, cls.CLEAN_AF_DIR, cls.ALIGNED_DIR,
            cls.REPORTS_DIR, cls.VALIDATED_DIR, cls.RESULTS_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def summary(cls):
        """Imprime un resumen de la configuración actual."""
        print("=" * 60)
        print("Configuración del proyecto")
        print("=" * 60)
        print(f"  PROJECT_ROOT:    {cls.PROJECT_ROOT}")
        print(f"  DATA_DIR:        {cls.DATA_DIR}")
        print(f"  SIFTS_FILE:      {cls.SIFTS_FILE}")
        print(f"  Workers:         {cls.DEFAULT_WORKERS}")
        print(f"  Min residues:    {cls.MIN_RESIDUES}")
        print(f"  Min common res:  {cls.MIN_COMMON_RESIDUES}")
        print("=" * 60)
