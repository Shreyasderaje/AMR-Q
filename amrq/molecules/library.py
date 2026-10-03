"""Compound library loading + validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .processing import parse_smiles
from rdkit import Chem

DEFAULT_LIBRARY = Path(__file__).resolve().parents[2] / "data" / "compounds" / "antibiotic_library.csv"

EXPECTED_COLUMNS = ["name", "smiles", "drug_class", "mechanism", "known_targets",
                    "inchikey", "source"]


def load_library(path: Path | str | None = None, limit: int | None = None,
                 categories: list[str] | None = None) -> pd.DataFrame:
    """Load the curated antibiotic library CSV.

    ``categories`` optionally filters by drug_class (substring match, case
    insensitive). ``limit`` caps the number of rows (applied after shuffle
    with a fixed seed for reproducibility).
    """
    path = Path(path) if path else DEFAULT_LIBRARY
    df = pd.read_csv(path).fillna({"known_targets": "", "mechanism": "",
                                   "inchikey": "", "source": ""})
    if categories:
        pattern = "|".join(categories)
        df = df[df["drug_class"].str.contains(pattern, case=False, na=False)]
    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)


def validate_library(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """RDKit-validate every SMILES row; drop invalid/duplicate entries.

    Returns (clean_df, dropped) where dropped carries the reason per row.
    """
    clean_rows, dropped, seen = [], [], set()
    for _, row in df.iterrows():
        mol = parse_smiles(str(row["smiles"]))
        if mol is None:
            dropped.append({"name": row["name"], "reason": "SMILES failed to parse"})
            continue
        try:
            key = Chem.MolToInchiKey(mol)
        except Exception:
            key = ""
        if not key:
            key = "smiles:" + Chem.MolToSmiles(mol)
        if key in seen:
            dropped.append({"name": row["name"], "reason": "duplicate compound"})
            continue
        seen.add(key)
        clean = dict(row)
        clean["inchikey"] = key
        clean_rows.append(clean)
    return pd.DataFrame(clean_rows, columns=df.columns), dropped
