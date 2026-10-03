"""Ligand-pocket interaction features (docking-lite).

AMR-Q does not run a full flexible docking search. Instead each ligand
conformer is rigidly placed into the pocket with a handful of candidate
orientations (centroid match + principal-axis alignment), and interaction
features are computed with a NeighborSearch over pocket heavy atoms:

- contact_frac            fraction of pocket residues within 4.5 A of the ligand
- hbond_contacts          ligand N/O/S within [2.4, 3.5] A of pocket N/O
- hydrophobic_contacts    ligand C within 4.5 A of pocket C/S
- electrostatic           charge-complementarity sum over pairs < 6 A
- clash                   ligand-pocket heavy-atom pairs closer than 2.6 A
- metal_proximity_score   Gaussian closeness of chelating atoms to bound metals

The best pose by a simple pre-score wins; its features feed the quantum
model and the ML ranker. This is deliberately simple, fast and transparent
-- the interesting science happens in the VQE + ML layers.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .processing import heavy_atom_mask, ligand_coords, polar_acceptor_mask

CLASH_DISTANCE = 2.6
CONTACT_DISTANCE = 4.5
HBOND_MIN, HBOND_MAX = 2.4, 3.5
ELECTRO_DISTANCE = 6.0


def _rotation_from_axes(lig_axes: np.ndarray, pocket_axes: np.ndarray,
                        sign1: int, sign2: int) -> np.ndarray:
    """Rotation aligning ligand principal axes onto pocket axes (with a sign
    flip on the first two pocket axes to probe opposite orientations)."""
    p = pocket_axes.copy()
    p[0] *= sign1
    p[1] *= sign2
    if np.linalg.det(p @ lig_axes.T) < 0:  # keep a proper rotation
        p[2] *= -1
    return p @ lig_axes.T


def candidate_poses(lig_coords: np.ndarray, pocket) -> list[np.ndarray]:
    """Generate candidate rigid placements of the ligand into the pocket."""
    lig_center = lig_coords.mean(axis=0)
    lig_centered = lig_coords - lig_center
    _, _, lig_axes = np.linalg.svd(lig_centered, full_matrices=False)
    pocket_centered = pocket.coords - pocket.centroid
    _, _, pocket_axes = np.linalg.svd(pocket_centered, full_matrices=False)

    poses = []
    # pose 0: pure centroid translation (no reorientation)
    poses.append(lig_centered + pocket.centroid)
    # pose 1-4: principal-axis alignments with sign variants
    for sign1, sign2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        R = _rotation_from_axes(lig_axes, pocket_axes, sign1, sign2)
        poses.append((R @ lig_centered.T).T + pocket.centroid)
    return poses


def pose_features(lig_coords: np.ndarray, lig_charges: np.ndarray,
                  heavy: np.ndarray, polar: np.ndarray,
                  chelating_indices: list[int], pocket,
                  pocket_tree: cKDTree) -> dict:
    """Compute interaction features for one rigid pose."""
    if len(lig_coords) == 0:
        raise ValueError("empty ligand coordinates")

    pocket_atoms = pocket.coords
    pocket_elements = pocket.elements
    pocket_is_polar = np.isin(pocket_elements, ["N", "O", "S"])
    pocket_is_carbon = pocket_elements == "C"

    contact_residues = set()
    hbond = 0
    hydrophobic = 0
    clash = 0
    electro = 0.0
    lig_heavy_idx = np.where(heavy)[0]

    for li in lig_heavy_idx:
        pos = lig_coords[li]
        for p_idx in pocket_tree.query_ball_point(pos, ELECTRO_DISTANCE):
            d = float(np.linalg.norm(pocket_atoms[p_idx] - pos))
            if d > ELECTRO_DISTANCE:
                continue
            if d < CONTACT_DISTANCE:
                contact_residues.add(int(pocket.atom_residue[p_idx]))
                if pocket_is_carbon[p_idx] or pocket_elements[p_idx] == "S":
                    hydrophobic += 1
            if HBOND_MIN <= d <= HBOND_MAX and polar[li] and pocket_is_polar[p_idx]:
                hbond += 1
            if d < CLASH_DISTANCE:
                clash += 1
            electro += float(lig_charges[li]) * float(
                _pocket_partial_charge(pocket_elements[p_idx])) / max(d, 1.5)

    contact_frac = len(contact_residues) / max(len(pocket.residues), 1)
    # favorable electrostatics: negative q_lig * q_pock sums -> complementarity
    electro_norm = float(np.tanh(-electro / 4.0))

    metal_score = 0.0
    if pocket.metal_coords is not None and len(pocket.metal_coords) > 0 and chelating_indices:
        dmin = min(
            float(np.min(np.linalg.norm(pocket.metal_coords - lig_coords[ci], axis=1)))
            for ci in chelating_indices if ci < len(lig_coords)
        )
        metal_score = float(np.exp(-((dmin - 2.3) ** 2) / 1.8))

    return {
        "contact_frac": round(contact_frac, 4),
        "n_contact_residues": len(contact_residues),
        "hbond_contacts": int(hbond),
        "hydrophobic_contacts": int(hydrophobic),
        "electrostatic_complementarity": round(electro_norm, 4),
        "clash": int(clash),
        "metal_proximity_score": round(metal_score, 4),
    }


_POCKET_CHARGE = {"N": -0.3, "O": -0.4, "S": -0.2, "C": 0.0}


def _pocket_partial_charge(element: str) -> float:
    """Coarse per-element partial charge for pocket heavy atoms."""
    return _POCKET_CHARGE.get(element, 0.0)


def pose_pre_score(feat: dict) -> float:
    """Simple pose-selection score (more contacts, fewer clashes win)."""
    return (feat["contact_frac"] + 0.05 * feat["hbond_contacts"]
            - 0.10 * min(feat["clash"], 5) / 5.0
            + 0.3 * feat["metal_proximity_score"])


def best_pose_features(mol, pocket, chelating_indices: list[int],
                       charges: np.ndarray | None = None) -> tuple[dict, int]:
    """Evaluate all candidate poses and return (best features, pose index)."""
    from .processing import gasteiger_charges

    if charges is None:
        charges = gasteiger_charges(mol)
    heavy = heavy_atom_mask(mol)
    polar = polar_acceptor_mask(mol)
    pocket_tree = cKDTree(pocket.coords)

    best_feat, best_idx, best_score = None, 0, -np.inf
    coords = ligand_coords(mol)
    for p_i, pose in enumerate(candidate_poses(coords, pocket)):
        feat = pose_features(pose, charges, heavy, polar, chelating_indices,
                             pocket, pocket_tree)
        score = pose_pre_score(feat)
        if score > best_score:
            best_feat, best_idx, best_score = feat, p_i, score
    if best_feat is None:
        raise ValueError("pose generation failed")
    return best_feat, best_idx
