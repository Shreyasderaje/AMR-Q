#!/usr/bin/env python
"""Build data/compounds/antibiotic_library.csv from library_source.json.

SMILES that are empty in the source file are looked up from PubChem (PUG
REST, name -> isomeric SMILES) with a ChEMBL fallback. Every SMILES is then
validated with RDKit (largest fragment kept), deduplicated by InChIKey, and
written to CSV. Entries that cannot be resolved are reported and skipped.

Usage:
    python scripts/build_library.py            # fetch + validate + write
    python scripts/build_library.py --offline  # only use SMILES already present
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

SOURCE = ROOT / "data" / "compounds" / "library_source.json"
OUT = ROOT / "data" / "compounds" / "antibiotic_library.csv"

PUBCHEM = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           "{name}/property/CanonicalSMILES,IsomericSMILES,ConnectivitySMILES/JSON")
CHEMBL_SEARCH = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={name}"


def pubchem_smiles(name: str) -> str | None:
    try:
        r = requests.get(PUBCHEM.format(name=requests.utils.quote(name)), timeout=20)
        if r.status_code != 200:
            return None
        props = r.json()["PropertyTable"]["Properties"][0]
        # PubChem renamed response keys (IsomericSMILES->SMILES,
        # CanonicalSMILES->ConnectivitySMILES); check all defensively.
        for key in ("SMILES", "IsomericSMILES", "ConnectivitySMILES",
                    "CanonicalSMILES"):
            smi = props.get(key)
            if smi and "." not in smi:
                return smi
        return props.get("SMILES") or props.get("IsomericSMILES") \
            or props.get("ConnectivitySMILES") or props.get("CanonicalSMILES")
    except Exception:
        return None


def chembl_smiles(name: str) -> str | None:
    try:
        r = requests.get(CHEMBL_SEARCH.format(name=requests.utils.quote(name)), timeout=20)
        if r.status_code != 200:
            return None
        mols = r.json().get("molecules", [])
        for mol in mols:
            structure = (mol.get("molecule_structures") or {})
            smi = structure.get("canonical_smiles")
            if smi and mol.get("molecule_type") == "Small molecule":
                return smi
    except Exception:
        return None
    return None


def resolve_smiles(entry: dict, offline: bool) -> tuple[str | None, str]:
    """Return (smiles, source) for one library entry."""
    if entry.get("smiles"):
        return entry["smiles"], "curated"
    if offline:
        return None, "skipped (offline)"
    name = entry["name"]
    smi = pubchem_smiles(name)
    if smi:
        return smi, "pubchem"
    time.sleep(0.2)
    smi = chembl_smiles(name)
    if smi:
        return smi, "chembl"
    return None, "not found in PubChem or ChEMBL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="do not query web APIs; only use existing SMILES")
    args = parser.parse_args()

    from rdkit import Chem

    from amrq.molecules.processing import parse_smiles

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows, skipped, seen = [], [], set()

    for i, entry in enumerate(source["entries"], 1):
        smiles, how = resolve_smiles(entry, args.offline)
        mol = parse_smiles(smiles) if smiles else None
        if mol is None:
            skipped.append((entry["name"], how if smiles is None else "SMILES invalid"))
            print(f"  [{i:3d}] SKIP  {entry['name']:34s} ({how if smiles else 'invalid'})")
            continue
        try:
            key = Chem.MolToInchiKey(mol)
        except Exception:
            key = "smiles:" + Chem.MolToSmiles(mol)
        if key in seen:
            skipped.append((entry["name"], "duplicate"))
            print(f"  [{i:3d}] DUP   {entry['name']}")
            continue
        seen.add(key)
        parent_smiles = Chem.MolToSmiles(mol)
        rows.append({
            "name": entry["name"],
            "smiles": parent_smiles,
            "drug_class": entry["drug_class"],
            "mechanism": entry["mechanism"],
            "known_targets": entry.get("known_targets", ""),
            "inchikey": key,
            "source": how if how in ("pubchem", "chembl", "curated") else "curated",
        })
        print(f"  [{i:3d}] OK    {entry['name']:34s} ({how})")
        time.sleep(0.25 if how in ("pubchem", "chembl") else 0.0)

    import csv

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "smiles", "drug_class",
                                                "mechanism", "known_targets",
                                                "inchikey", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} compounds -> {OUT}")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for name, why in skipped:
            print(f"  - {name}: {why}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
