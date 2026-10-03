"""Tests for the molecules layer (RDKit, library, interaction features)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amrq.molecules.interactions import best_pose_features  # noqa: E402
from amrq.molecules.library import load_library, validate_library  # noqa: E402
from amrq.molecules.processing import (chelating_atoms, descriptors,  # noqa: E402
                                       gasteiger_charges, mol_to_svg,
                                       parse_smiles, prepare_molecule)
from amrq.protein.parser import load_structure, pocket_from_residues  # noqa: E402

from test_protein import make_synthetic_pdb  # noqa: E402


def test_parse_smiles_basic_and_salts():
    mol = parse_smiles("CCO")
    assert mol is not None and mol.GetNumAtoms() == 3
    # sodium salt -> parent acid kept
    mol = parse_smiles("CC(=O)[O-].[Na+]")
    assert mol is not None
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    assert "Na" not in symbols
    assert parse_smiles("not_a_smiles!!") is None
    assert parse_smiles("") is None


def test_prepare_molecule_produces_3d_conformer():
    mol = prepare_molecule("c1ccccc1C(=O)O", seed=42)
    assert mol is not None
    conf = mol.GetConformer()
    pos = np.array(conf.GetPositions())
    assert pos.shape[1] == 3
    assert np.abs(pos).max() > 0.5  # real 3D geometry, not all-zero


def test_descriptors_caffeine():
    mol = prepare_molecule("Cn1c(=O)c2c(ncn2C)n(C)c1=O", seed=1)
    d = descriptors(mol)
    assert d["heavy_atoms"] == 14
    assert 194.0 < d["mw"] < 195.0
    assert d["formal_charge"] == 0
    assert d["max_negative_charge"] < -0.2   # carbonyl oxygens
    assert d["max_positive_charge"] > 0.1


def test_chelating_atoms_detection():
    captopril = parse_smiles("C[C@H](CS)C(=O)N1CCCC1C(=O)O")
    idx, names = chelating_atoms(captopril)
    assert len(idx) >= 3  # thiol S + two carboxyl/acid O
    assert any("thiol" in n for n in names)
    benzoic = parse_smiles("c1ccccc1C(=O)O")
    idx, names = chelating_atoms(benzoic)
    assert len(idx) >= 2


def test_gasteiger_charges_sum_to_formal_charge():
    mol = parse_smiles("CC(=O)O")
    mol = __import__("rdkit").Chem.AddHs(mol)
    q = gasteiger_charges(mol)
    assert q.shape[0] == mol.GetNumAtoms()
    assert abs(q.sum() - 0.0) < 1e-3  # neutral molecule


def test_mol_to_svg():
    mol = parse_smiles("CCOc1ccc2nc(sc2c1)C(=O)O")  # some ring system
    svg = mol_to_svg(mol)
    assert svg.startswith("<svg") and "path" in svg or "<rect" in svg


def test_best_pose_features_against_synthetic_pocket():
    structure = load_structure(make_synthetic_pdb(), structure_id="t")
    pocket = pocket_from_residues(structure, "A", [10, 11, 12])
    mol = prepare_molecule("CC(=O)Oc1ccccc1C(=O)O", seed=3)  # aspirin
    charges = gasteiger_charges(mol)
    chel_idx, _ = chelating_atoms(mol)
    feat, pose_idx = best_pose_features(mol, pocket, chel_idx, charges)
    assert feat["contact_frac"] > 0.0
    assert feat["n_contact_residues"] >= 1
    assert 0.0 <= feat["metal_proximity_score"] <= 1.0
    assert feat["clash"] >= 0


def test_library_loads_and_validates():
    df = load_library()
    assert len(df) >= 140
    assert {"name", "smiles", "drug_class", "mechanism", "known_targets"} <= set(df.columns)
    classes = set(df["drug_class"])
    assert "beta-lactamase inhibitor" in classes
    assert "decoy" in classes
    clean, dropped = validate_library(df)
    assert len(clean) >= 140
    assert len(dropped) == 0
    assert clean["inchikey"].str.len().ge(10).all()


def test_library_limit_is_deterministic():
    a = load_library(limit=30)
    b = load_library(limit=30)
    assert list(a["name"]) == list(b["name"])
    assert len(a) == 30


def test_known_targets_metadata_present():
    df = load_library()
    avibactam = df[df["name"] == "Avibactam"].iloc[0]
    assert "class_a" in avibactam["known_targets"]
    captopril = df[df["name"] == "Captopril"].iloc[0]
    assert "class_b" in captopril["known_targets"]
    ibu = df[df["name"] == "Ibuprofen"].iloc[0]
    assert ibu["known_targets"] == ""
