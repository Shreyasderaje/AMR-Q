"""Feature vector definition shared by training, inference and export.

The ranker fuses three feature families into one vector:
  1. ligand physicochemical descriptors (RDKit)
  2. ligand-pocket interaction features (docking-lite poses)
  3. quantum features from the VQE binding model (stabilization energy,
     charge-transfer character)
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "mw",                        # molecular weight
    "crippen_logp",              # hydrophobicity
    "tpsa",                      # topological polar surface area
    "hbd",                       # H-bond donors
    "hba",                       # H-bond acceptors
    "rotatable_bonds",
    "heavy_atoms",
    "aromatic_rings",
    "fsp3",
    "formal_charge",
    "max_positive_charge",
    "max_negative_charge",
    "chelating_atoms",
    "radius_gyration",
    # interaction features (best pose)
    "contact_frac",
    "hbond_contacts",
    "hydrophobic_contacts",
    "electrostatic_complementarity",
    "clash",
    "metal_proximity_score",
    "pocket_polar_frac",
    # quantum features (VQE on the 4-qubit binding model)
    "quantum_delta_e",           # E_coupled - E_decoupled, eV (<=0 stabilizing)
    "charge_transfer_weight",    # ground-state weight on CT configurations
]

FEATURE_NAMES = list(FEATURE_NAMES)
N_FEATURES = len(FEATURE_NAMES)


def build_feature_vector(ligand: dict, interaction: dict, pocket_polar_frac: float,
                         quantum: dict) -> np.ndarray:
    """Assemble the ordered feature vector from the three feature dicts."""
    values = {
        "mw": ligand.get("mw", 0.0),
        "crippen_logp": ligand.get("crippen_logp", 0.0),
        "tpsa": ligand.get("tpsa", 0.0),
        "hbd": ligand.get("hbd", 0),
        "hba": ligand.get("hba", 0),
        "rotatable_bonds": ligand.get("rotatable_bonds", 0),
        "heavy_atoms": ligand.get("heavy_atoms", 0),
        "aromatic_rings": ligand.get("aromatic_rings", 0),
        "fsp3": ligand.get("fsp3", 0.0),
        "formal_charge": ligand.get("formal_charge", 0),
        "max_positive_charge": ligand.get("max_positive_charge", 0.0),
        "max_negative_charge": ligand.get("max_negative_charge", 0.0),
        "chelating_atoms": ligand.get("chelating_atoms", 0),
        "radius_gyration": ligand.get("radius_gyration", 0.0),
        "contact_frac": interaction.get("contact_frac", 0.0),
        "hbond_contacts": interaction.get("hbond_contacts", 0),
        "hydrophobic_contacts": interaction.get("hydrophobic_contacts", 0),
        "electrostatic_complementarity": interaction.get("electrostatic_complementarity", 0.0),
        "clash": interaction.get("clash", 0),
        "metal_proximity_score": interaction.get("metal_proximity_score", 0.0),
        "pocket_polar_frac": pocket_polar_frac,
        "quantum_delta_e": quantum.get("delta_e", 0.0),
        "charge_transfer_weight": quantum.get("ct_weight", 0.0),
    }
    return np.array([float(values[name]) for name in FEATURE_NAMES], dtype=float)


def feature_dict_from(ligand: dict, interaction: dict, pocket_polar_frac: float,
                      quantum: dict) -> dict:
    """Same content as a plain dict (used by the expert scorer / attribution)."""
    vec = build_feature_vector(ligand, interaction, pocket_polar_frac, quantum)
    return dict(zip(FEATURE_NAMES, vec.tolist()))
