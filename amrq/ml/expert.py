"""Expert rule scorer - weak supervision for the ML ranker.

Training a binding-affinity predictor without experimental data would be
dishonest. AMR-Q is explicit about what it does instead: the training labels
come from a transparent medicinal-chemistry rule ensemble (contact and
H-bond coverage, charge complementarity, metal chelation for metallo-enzymes,
clash penalties, quantum stabilization, plus a documented boost when a
compound is *known* to bind/inhibit the target's resistance class). The ML
ranker distills these labels into a smooth function of the raw features, and
the same expert scorer doubles as the no-torch fallback.

`z_delta_e` is the run-level z-score of the quantum stabilization energy, so
the quantum term is comparable across targets and runs.
"""
from __future__ import annotations

import math


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _known_set(known_targets) -> set:
    """Accept either a 'class_a;class_c' string or a list of class codes."""
    if isinstance(known_targets, (list, tuple)):
        return set(str(t).strip() for t in known_targets if str(t).strip())
    return set(t.strip() for t in str(known_targets or "").split(";") if t.strip())


def expert_score(feat: dict, target_resistance_class: str = "",
                 known_targets="", z_delta_e: float = 0.0,
                 chelation_relevant: bool = False) -> float:
    """Resistance-breaking potential in [0, 1] from features + priors."""
    known = _known_set(known_targets)

    pre = (
        0.28 * float(feat.get("contact_frac", 0.0))
        + 0.12 * _sigmoid((float(feat.get("hbond_contacts", 0)) - 0.5) / 1.2)
        + 0.10 * _sigmoid((float(feat.get("hydrophobic_contacts", 0)) - 6.0) / 6.0)
        + 0.08 * max(float(feat.get("electrostatic_complementarity", 0.0)), 0.0)
        - 0.20 * min(float(feat.get("clash", 0)) / 3.0, 1.0)
        + 0.12 * _sigmoid(-z_delta_e / 0.8)          # quantum stabilization
    )
    if chelation_relevant:
        pre += 0.10 * float(feat.get("metal_proximity_score", 0.0))
        pre += 0.06 * _sigmoid((float(feat.get("chelating_atoms", 0)) - 1.0) / 1.0)
    if target_resistance_class and target_resistance_class in known:
        pre += 0.30                                   # documented prior: known binder
    return max(0.0, min(1.0, pre))


def expert_components(feat: dict, target_resistance_class: str = "",
                      known_targets="", z_delta_e: float = 0.0,
                      chelation_relevant: bool = False) -> dict:
    """Per-term contributions (for explainability when torch is absent)."""
    known = _known_set(known_targets)
    comps = {
        "contacts": 0.28 * float(feat.get("contact_frac", 0.0)),
        "hbonds": 0.12 * _sigmoid((float(feat.get("hbond_contacts", 0)) - 0.5) / 1.2),
        "hydrophobic": 0.10 * _sigmoid((float(feat.get("hydrophobic_contacts", 0)) - 6.0) / 6.0),
        "electrostatics": 0.08 * max(float(feat.get("electrostatic_complementarity", 0.0)), 0.0),
        "clash": -0.20 * min(float(feat.get("clash", 0)) / 3.0, 1.0),
        "quantum": 0.12 * _sigmoid(-z_delta_e / 0.8),
    }
    if chelation_relevant:
        comps["metal_chelation"] = (
            0.10 * float(feat.get("metal_proximity_score", 0.0))
            + 0.06 * _sigmoid((float(feat.get("chelating_atoms", 0)) - 1.0) / 1.0))
    if target_resistance_class and target_resistance_class in known:
        comps["known_activity_prior"] = 0.30
    return comps
