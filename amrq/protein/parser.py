"""Protein structure parsing and binding-pocket detection (BioPython).

A "pocket" is a set of residues plus the heavy atoms of those residues. AMR-Q
supports four ways to define one:

1. ``curated``  - known catalytic/pocket residues for a named resistance
                  protein (numbering as in the PDB entry, usually Ambler).
2. ``metal``    - residues around bound metal cofactors (e.g. the Zn site of
                  metallo-beta-lactamases such as NDM-1).
3. ``cocrystal``- residues around a co-crystallized ligand (works for any
                  uploaded PDB that has a ligand).
4. ``manual``   - caller-specified chain + residue numbers.
"""
from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser

# common crystallization additives / ions that are NOT real binding ligands
HETATM_EXCLUDE = {
    "HOH", "DOD", "SO4", "PO4", "P6G", "PG4", "PGE", "1PE", "GOL", "EDO", "BME",
    "DTT", "TCE", "ACT", "ACY", "MES", "TRS", "EPE", "POP", "FLC", "CIT", "MPD",
    "PEG", "IPA", "ETOH", "ACETONE", "ZN", "MG", "CA", "NA", "CL", "K", "MN",
    "FE", "NI", "CU", "CD", "HG", "CO", "CS", "RB", "YB", "SE4", "O4B", "IOD",
    "NO3", "NH4", "F", "BR", "I", "TAU", "EUC", "BEN", "DMS", "MLI", "CO3",
}
METAL_RESNAMES = {"ZN", "MG", "MN", "FE", "NI", "CU", "CO", "CD", "HG", "CA", "NA", "K"}
POLAR_ELEMENTS = {"N", "O", "S"}


@dataclass
class Pocket:
    """Binding pocket: residues + their heavy atoms + geometry summary."""

    method: str                                  # curated | metal | cocrystal | manual
    chain: str
    residues: list[tuple[str, int, str]]         # (chain, resseq, resname)
    coords: np.ndarray                           # (N, 3) heavy-atom coordinates
    elements: np.ndarray                         # (N,) element symbols
    atom_residue: np.ndarray                     # (N,) index into residues
    centroid: np.ndarray
    axes: np.ndarray                             # (3, 3) principal axes (rows)
    polar_frac: float
    hydrophobic_frac: float
    metal_coords: np.ndarray | None = None       # (M, 3) bound metal ions
    metal_names: list[str] = field(default_factory=list)
    missing_curated: list[int] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.coords)

    def summary(self) -> dict:
        return {
            "method": self.method,
            "chain": self.chain,
            "n_residues": len(self.residues),
            "residues": [f"{res}{num}" for (_, num, res) in self.residues],
            "n_atoms": self.n_atoms,
            "polar_frac": round(self.polar_frac, 3),
            "hydrophobic_frac": round(self.hydrophobic_frac, 3),
            "metals": self.metal_names,
            "missing_curated": self.missing_curated,
            "centroid": [round(float(v), 2) for v in self.centroid],
        }


def load_structure(source: str | Path, structure_id: str = "amrq") -> object:
    """Load a Bio.PDB Structure from a file path or raw PDB text."""
    parser = PDBParser(QUIET=True)
    path = Path(source)
    if path.exists():
        return parser.get_structure(structure_id, str(path))
    text = str(source)
    if "ATOM" not in text and "HETATM" not in text:
        raise ValueError("Not a PDB file path and no ATOM/HETATM records found in input")
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False, encoding="utf-8") as fh:
        fh.write(text)
        tmp = Path(fh.name)
    try:
        return parser.get_structure(structure_id, str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


def _residue_atoms(residue) -> tuple[list, list[str], list[str]]:
    coords, elements, names = [], [], []
    for atom in residue.get_unpacked_list():
        if atom.element and atom.element.upper() == "H":
            continue
        coords.append(atom.coord.astype(float))
        elements.append((atom.element or atom.get_name()[0]).upper())
        names.append(atom.get_name())
    return coords, elements, names


def _build_pocket(method: str, chain: str, residues: list,
                  metal_coords: np.ndarray | None = None,
                  metal_names: list[str] | None = None,
                  missing_curated: list[int] | None = None) -> Pocket:
    all_coords, all_elements, all_res_idx = [], [], []
    for r_i, res in enumerate(residues):
        coords, elements, _ = _residue_atoms(res)
        all_coords.extend(coords)
        all_elements.extend(elements)
        all_res_idx.extend([r_i] * len(coords))
    coords = np.array(all_coords)
    elements = np.array(all_elements)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    # principal axes via SVD (rows = axes)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh
    polar = float(np.isin(elements, list(POLAR_ELEMENTS)).mean())
    hydro = float((elements == "C").mean())
    return Pocket(
        method=method, chain=chain,
        residues=[(res.get_parent().id, res.id[1], res.get_resname()) for res in residues],
        coords=coords, elements=elements, atom_residue=np.array(all_res_idx),
        centroid=centroid, axes=axes, polar_frac=polar, hydrophobic_frac=hydro,
        metal_coords=metal_coords, metal_names=metal_names or [],
        missing_curated=missing_curated or [],
    )


def pocket_from_residues(structure, chain_id: str, resnums: list[int]) -> Pocket:
    """Curated/manual pocket: named residues of one chain."""
    model = next(structure.get_models())
    chain = model[chain_id] if chain_id in [c.id for c in model] else next(model.get_chains())
    wanted = set(int(r) for r in resnums)
    found = [res for res in chain.get_residues() if res.id[1] in wanted]
    missing = sorted(wanted - {res.id[1] for res in found})
    if len(found) < 3:
        raise ValueError(
            f"Only {len(found)} of the requested residues exist in chain {chain.id} "
            f"(missing: {missing}). Check chain ID and residue numbering.")
    return _build_pocket("curated", chain.id, found, missing_curated=missing)


def pocket_from_cocrystal(structure, radius: float = 7.0) -> Pocket:
    """Pocket = residues within ``radius`` of the largest co-crystallized ligand."""
    model = next(structure.get_models())
    ligands = []
    for res in model.get_residues():
        if res.id[0] == " " or res.get_resname() in HETATM_EXCLUDE:
            continue
        coords, elements, _ = _residue_atoms(res)
        if len(coords) >= 4:
            ligands.append((len(coords), res, np.array(coords)))
    if not ligands:
        raise ValueError("No suitable co-crystallized ligand found (only water/ions/additives).")
    ligands.sort(key=lambda t: -t[0])
    _, lig_res, lig_coords = ligands[0]
    lig_center = lig_coords.mean(axis=0)

    protein_atoms, protein_residues = [], []
    for res in model.get_residues():
        if res.id[0] != " " or res.get_resname() == "HOH":
            continue
        coords, elements, _ = _residue_atoms(res)
        protein_atoms.extend(coords)
        protein_residues.extend([res] * len(coords))
    if not protein_atoms:
        raise ValueError("Structure contains no protein atoms.")
    protein_coords = np.array(protein_atoms)
    tree = cKDTree(protein_coords)
    hit_residues = {protein_residues[i]
                    for i in tree.query_ball_point(lig_center, radius)}
    hits = [r for r in hit_residues if r.get_resname() not in HETATM_EXCLUDE]
    if len(hits) < 3:
        raise ValueError("Too few protein residues near the co-crystal ligand.")
    return _build_pocket("cocrystal", lig_res.get_parent().id, hits)


def pocket_from_metal(structure, radius: float = 6.5,
                      metals: set[str] | None = None) -> Pocket:
    """Pocket = residues chelating one metal site (e.g. the Zn pair of NDM-1).

    Metal ions are clustered into sites (single-linkage, 5 A); the largest
    site defines the pocket, which keeps homodimer structures (two identical
    active sites) from mixing residues of both chains.
    """
    metals = metals or {"ZN", "MG", "MN", "FE", "NI", "CU", "CO"}
    model = next(structure.get_models())
    metal_atoms, metal_names = [], []
    for res in model.get_residues():
        if res.id[0] == " " or res.get_resname() not in metals:
            continue
        for atom in res.get_unpacked_list():
            metal_atoms.append(atom.coord.astype(float))
            metal_names.append(res.get_resname())
    if not metal_atoms:
        raise ValueError("No bound metal ions found in structure.")
    metal_coords = np.array(metal_atoms)

    # cluster metal ions into sites
    sites: list[list[np.ndarray]] = []
    for m in metal_coords:
        for site in sites:
            if any(np.linalg.norm(m - s) < 5.0 for s in site):
                site.append(m)
                break
        else:
            sites.append([m])
    sites.sort(key=len, reverse=True)
    site_coords = np.array(sites[0])

    protein_atoms, protein_residues = [], []
    for res in model.get_residues():
        if res.id[0] != " " or res.get_resname() == "HOH":
            continue
        coords, elements, _ = _residue_atoms(res)
        protein_atoms.extend(coords)
        protein_residues.extend([res] * len(coords))
    if not protein_atoms:
        raise ValueError("Structure contains no protein atoms.")
    protein_coords = np.array(protein_atoms)
    tree = cKDTree(protein_coords)
    by_key: dict[tuple, object] = {}
    for m in site_coords:
        for i in tree.query_ball_point(m, radius):
            res = protein_residues[i]
            key = (res.get_parent().id, res.id[1])
            by_key.setdefault(key, res)
    hits = sorted(by_key.values(), key=lambda r: r.id[1])
    chain = hits[0].get_parent().id if hits else "?"
    return _build_pocket("metal", chain, hits,
                         metal_coords=site_coords,
                         metal_names=sorted(set(metal_names)))


def detect_pocket(structure, method: str, chain: str = "A",
                  residues: list[int] | None = None) -> Pocket:
    """Dispatch pocket detection; curated falls back to metal then cocrystal."""
    method = (method or "cocrystal").lower()
    if method in ("curated", "manual") and residues:
        try:
            return pocket_from_residues(structure, chain, residues)
        except ValueError:
            if method == "manual":
                raise
    if method in ("curated", "metal"):
        try:
            return pocket_from_metal(structure)
        except ValueError:
            pass
    return pocket_from_cocrystal(structure)
