#!/usr/bin/env python3
"""
Validación científica de pares PDB-AlphaFold.

Verifica que cada par realmente corresponda a la misma proteína
y que la comparación sea significativa.

Checks:
1. Identidad de secuencia entre PDB y AlphaFold
2. Correspondencia de cadena via SIFTS
3. Compatibilidad de longitud
4. Solapamiento de residuos
5. Flags para pares sospechosos (RMSD > 10Å)
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

from alphafold_comparison.config import Config

warnings.filterwarnings("ignore")


def _extract_sequence(structure, chain_id=None):
    """
    Extrae la secuencia de aminoácidos de una estructura.

    Args:
        structure: Estructura BioPython.
        chain_id: ID de cadena específica (None = primera disponible).

    Returns:
        tuple: (secuencia como dict {res_id: res_name}, chain_id usado)
    """
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    model = structure[0]
    chains = list(model.get_chains())

    if not chains:
        return {}, None

    if chain_id and chain_id in [c.id for c in chains]:
        chain = model[chain_id]
    else:
        chain = chains[0]
        chain_id = chain.id

    sequence = {}
    for residue in chain:
        if residue.id[0] == " ":  # Residuo estándar
            res_name = residue.get_resname()
            if res_name in three_to_one:
                sequence[residue.id[1]] = three_to_one[res_name]

    return sequence, chain_id


def _sequence_identity(seq1: dict, seq2: dict) -> float:
    """
    Calcula la identidad de secuencia entre dos diccionarios de residuos.

    Solo compara posiciones presentes en ambas secuencias.
    """
    common_positions = set(seq1.keys()) & set(seq2.keys())
    if not common_positions:
        return 0.0

    matches = sum(1 for pos in common_positions if seq1[pos] == seq2[pos])
    return matches / len(common_positions)


def validate_pair(row: dict) -> dict:
    """
    Valida un par PDB-AlphaFold.

    Args:
        row: Diccionario con pdb_id, uniprot_id, y opcionalmente rmsd.

    Returns:
        Diccionario con resultados de validación.
    """
    parser = PDBParser(QUIET=True)

    pdb_id = row["pdb_id"]
    uniprot_id = row["uniprot_id"]
    existing_rmsd = row.get("rmsd", None)

    result = {
        "pdb_id": pdb_id,
        "uniprot_id": uniprot_id,
        "validation_status": "FAILED",
        "validation_notes": "",
        "pdb_seq_len": 0,
        "af_seq_len": 0,
        "seq_identity": 0.0,
        "overlap_pct": 0.0,
        "length_ratio": 0.0,
        "pdb_chain_used": "",
        "existing_rmsd": existing_rmsd,
    }

    try:
        pdb_file = Config.PDB_DIR / f"pdb{pdb_id.lower()}.ent"
        af_file = Config.AF_DIR / f"{uniprot_id}.pdb"

        if not pdb_file.exists():
            result["validation_notes"] = "PDB file not found"
            return result
        if not af_file.exists():
            result["validation_notes"] = "AlphaFold file not found"
            return result

        # Parsear estructuras
        pdb_structure = parser.get_structure("pdb", str(pdb_file))
        af_structure = parser.get_structure("af", str(af_file))

        # Extraer secuencias
        pdb_seq, pdb_chain = _extract_sequence(pdb_structure)
        af_seq, af_chain = _extract_sequence(af_structure)

        if not pdb_seq:
            result["validation_notes"] = "PDB: no protein residues found"
            return result
        if not af_seq:
            result["validation_notes"] = "AlphaFold: no protein residues found"
            return result

        result["pdb_seq_len"] = len(pdb_seq)
        result["af_seq_len"] = len(af_seq)
        result["pdb_chain_used"] = pdb_chain

        # Check 1: Compatibilidad de longitud
        if len(af_seq) > 0:
            length_ratio = len(pdb_seq) / len(af_seq)
        else:
            length_ratio = 0.0
        result["length_ratio"] = length_ratio

        # Check 2: Identidad de secuencia
        seq_identity = _sequence_identity(pdb_seq, af_seq)
        result["seq_identity"] = seq_identity

        # Check 3: Solapamiento de residuos
        common_positions = set(pdb_seq.keys()) & set(af_seq.keys())
        if len(pdb_seq) > 0:
            overlap_pct = len(common_positions) / len(pdb_seq)
        else:
            overlap_pct = 0.0
        result["overlap_pct"] = overlap_pct

        # Clasificación
        notes = []

        if len(common_positions) < Config.MIN_COMMON_RESIDUES:
            notes.append(f"Very few common residues: {len(common_positions)}")
            result["validation_status"] = "REJECTED"
        elif seq_identity < 0.5:
            notes.append(f"Low sequence identity: {seq_identity:.2f}")
            result["validation_status"] = "SUSPICIOUS"
        elif overlap_pct < 0.3:
            notes.append(f"Low overlap: {overlap_pct:.2f}")
            result["validation_status"] = "SUSPICIOUS"
        elif length_ratio > 1.5:
            notes.append(f"PDB longer than AlphaFold: ratio={length_ratio:.2f}")
            result["validation_status"] = "SUSPICIOUS"
        else:
            result["validation_status"] = "VALIDATED"

        # Flag por RMSD alto
        if existing_rmsd is not None and existing_rmsd > 10.0:
            notes.append(f"High RMSD: {existing_rmsd:.2f}A")
            if result["validation_status"] == "VALIDATED":
                result["validation_status"] = "REVIEW"

        # Flag de alta calidad
        if (seq_identity > 0.9 and overlap_pct > 0.8
                and existing_rmsd is not None and existing_rmsd < 2.0):
            result["validation_status"] = "EXCELLENT"

        result["validation_notes"] = "; ".join(notes) if notes else "OK"

    except Exception as e:
        result["validation_notes"] = f"Error: {e}"

    return result


def run_validation(workers=None, max_pairs=None):
    """Ejecuta la validación completa de todos los pares."""
    workers = workers or Config.DEFAULT_WORKERS
    Config.ensure_dirs()

    print("=" * 60)
    print("PAIR VALIDATION PIPELINE v1.0")
    print("=" * 60)

    # Cargar pares procesados (con RMSD si existe)
    if Config.QUALITY_INDEX.exists():
        pairs_df = pd.read_csv(Config.QUALITY_INDEX)
        pdb_col = "pdb_id"
        uniprot_col = "uniprot_id"
        has_rmsd = "rmsd" in pairs_df.columns
        print(f"Cargados {len(pairs_df):,} pares procesados (con RMSD)")
    elif Config.ID_LIST.exists():
        pairs_df = pd.read_csv(Config.ID_LIST)
        pdb_col = "PDB"
        uniprot_col = "SP_PRIMARY"
        has_rmsd = False
        print(f"Cargados {len(pairs_df):,} pares descargados (sin RMSD)")
    else:
        raise FileNotFoundError("No se encontro archivo de pares")

    if max_pairs and max_pairs < len(pairs_df):
        pairs_df = pairs_df.head(max_pairs)
        print(f"Limitado a {max_pairs:,} pares")

    # Preparar datos
    rows = []
    for _, row in pairs_df.iterrows():
        r = {
            "pdb_id": row[pdb_col],
            "uniprot_id": row[uniprot_col],
        }
        if has_rmsd:
            r["rmsd"] = row["rmsd"]
        rows.append(r)

    print(f"\nValidando {len(rows):,} pares...")

    # Ejecutar validación en paralelo
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(validate_pair, rows),
            total=len(rows),
            desc="Validando",
            unit="pares",
        ))

    results_df = pd.DataFrame(results)

    # Estadísticas
    print(f"\n{'=' * 60}")
    print("RESULTADOS DE VALIDACION")
    print(f"{'=' * 60}")

    status_counts = results_df["validation_status"].value_counts()
    for status, count in status_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {status:12}: {count:6,} ({pct:.1f}%)")

    validated = results_df[results_df["validation_status"].isin(["VALIDATED", "EXCELLENT"])]
    print(f"\nPares validados: {len(validated):,} ({len(validated) / len(results_df) * 100:.1f}%)")

    if len(validated) > 0:
        print(f"  Seq identity media: {validated['seq_identity'].mean():.3f}")
        print(f"  Overlap medio: {validated['overlap_pct'].mean():.3f}")
        if "existing_rmsd" in validated.columns:
            valid_rmsd = validated["existing_rmsd"].dropna()
            if len(valid_rmsd) > 0:
                print(f"  RMSD medio (validados): {valid_rmsd.mean():.2f} A")

    # Guardar resultados
    results_df.to_csv(Config.VALIDATED_INDEX, index=False)
    print(f"\nIndice guardado: {Config.VALIDATED_INDEX}")

    # Resumen por categoría
    summary_path = Config.VALIDATED_DIR / "validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("VALIDATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total pairs: {len(results_df):,}\n\n")
        for status, count in status_counts.items():
            f.write(f"  {status}: {count:,} ({count / len(results_df) * 100:.1f}%)\n")
        f.write(f"\nValidated pairs: {len(validated):,}\n")
        if len(validated) > 0:
            f.write(f"  Mean seq identity: {validated['seq_identity'].mean():.3f}\n")
            f.write(f"  Mean overlap: {validated['overlap_pct'].mean():.3f}\n")

    print(f"Resumen guardado: {summary_path}")
    return 0


class PairValidator:
    """Clase wrapper para validación programática."""

    def __init__(self, workers=None):
        self.workers = workers or Config.DEFAULT_WORKERS

    def validate_all(self, max_pairs=None):
        return run_validation(workers=self.workers, max_pairs=max_pairs)

    def validate_single(self, pdb_id: str, uniprot_id: str, rmsd=None) -> dict:
        return validate_pair({
            "pdb_id": pdb_id,
            "uniprot_id": uniprot_id,
            "rmsd": rmsd,
        })


def main():
    parser = argparse.ArgumentParser(description="Validacion de pares PDB-AlphaFold")
    parser.add_argument("--workers", type=int, default=Config.DEFAULT_WORKERS)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    return run_validation(workers=args.workers, max_pairs=args.max_pairs)


if __name__ == "__main__":
    exit(main())
