"""Tests for protein parsing / pocket detection (no network required)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amrq.protein.fetcher import validate_pdb_id  # noqa: E402
from amrq.protein.parser import (detect_pocket, load_structure,  # noqa: E402
                                 pocket_from_cocrystal, pocket_from_metal,
                                 pocket_from_residues)
from amrq.protein.pockets import load_targets  # noqa: E402


def atom_line(serial, name, resname, chain, resseq, x, y, z, element,
              hetatm=False, occup=1.0):
    rec = "HETATM" if hetatm else "ATOM  "
    return (f"{rec}{serial:5d} {name:<4s} {resname:>3s} {chain}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occup:6.2f}{0.0:6.2f}          {element:>2s}")


def make_synthetic_pdb():
    """Small structure: 3 'pocket' residues near origin + a ligand + a Zn ion."""
    lines = []
    serial = 1

    def residue(resname, resseq, cx, cy, cz):
        nonlocal serial
        for (name, dx, dy, dz, el) in [
            ("N", 1.2, 0.0, 0.0, "N"), ("CA", 0.0, 0.0, 0.0, "C"),
            ("C", -1.2, 1.1, 0.0, "C"), ("O", -1.2, 2.3, 0.0, "O"),
        ]:
            lines.append(atom_line(serial, name, resname, "A", resseq,
                                   cx + dx, cy + dy, cz + dz, el))
            serial += 1

    residue("PHE", 10, 0.0, 0.0, 0.0)
    residue("HIS", 11, 3.0, 1.0, 0.5)
    residue("ASP", 12, 1.5, -3.0, 0.5)
    # filler residue far away
    residue("GLY", 20, 40.0, 40.0, 40.0)
    # co-crystal ligand near origin
    for i, (dx, dy, dz, el) in enumerate([(1.0, 1.0, 1.0, "C"), (2.0, 1.4, 0.5, "C"),
                                          (1.5, 2.0, 1.5, "O"), (0.2, 1.8, 0.2, "N")]):
        lines.append(atom_line(serial, f"C{i + 1}", "LIG", "A", 200,
                               dx, dy, dz, el, hetatm=True))
        serial += 1
    # zinc ion near the histidine
    lines.append(atom_line(serial, "ZN", "ZN", "A", 300, 3.0, 1.0, 2.0, "ZN",
                           hetatm=True))
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def structure():
    return load_structure(make_synthetic_pdb(), structure_id="test")


def test_validate_pdb_id():
    assert validate_pdb_id(" 1btl ") == "1BTL"
    with pytest.raises(ValueError):
        validate_pdb_id("nope!")
    with pytest.raises(ValueError):
        validate_pdb_id("12345")


def test_load_structure_from_text(structure):
    n_atoms = sum(1 for _ in structure.get_atoms())
    assert n_atoms >= 20


def test_pocket_from_residues(structure):
    pocket = pocket_from_residues(structure, "A", [10, 11, 12])
    assert len(pocket.residues) == 3
    assert pocket.summary()["residues"] == ["PHE10", "HIS11", "ASP12"]
    centroid = pocket.centroid
    assert np.linalg.norm(centroid) < 5.0  # cluster near origin
    assert 0.0 < pocket.polar_frac < 1.0


def test_pocket_from_residues_missing_raises(structure):
    with pytest.raises(ValueError):
        pocket_from_residues(structure, "A", [10, 11, 9999])


def test_pocket_from_cocrystal(structure):
    pocket = pocket_from_cocrystal(structure, radius=6.0)
    names = [r for _, _, r in pocket.residues]
    assert "LIG" not in names
    assert "GLY" not in names          # far residue excluded
    assert len(pocket.residues) >= 2   # near residues included
    assert pocket.method == "cocrystal"


def test_pocket_from_metal(structure):
    pocket = pocket_from_metal(structure, radius=4.0)
    assert pocket.metal_names == ["ZN"]
    names = [r for _, _, r in pocket.residues]
    assert "HIS" in names
    assert "GLY" not in names


def test_detect_pocket_fallback_chain(structure):
    pocket = detect_pocket(structure, "cocrystal")
    assert pocket.method == "cocrystal"


def test_curated_targets_load():
    targets = load_targets()
    assert len(targets) >= 5
    tem = targets["tem-1"]
    assert tem.pdb_id == "1BTL"
    assert 70 in tem.pocket_residues
    assert tem.resistance_class == "class_a"
    ndm = targets["ndm-1"]
    assert ndm.chelation_relevant is True
    assert ndm.pocket_method == "metal"
