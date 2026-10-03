"""Tests for the ML layer: features, expert scorer, model round-trip."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amrq.ml.expert import expert_components, expert_score  # noqa: E402
from amrq.ml.features import FEATURE_NAMES, N_FEATURES, build_feature_vector  # noqa: E402

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def base_features():
    lig = dict(mw=350.0, crippen_logp=1.5, tpsa=90.0, hbd=2, hba=5,
               rotatable_bonds=4, heavy_atoms=24, aromatic_rings=1,
               fsp3=0.3, formal_charge=0, max_positive_charge=0.2,
               max_negative_charge=0.45, chelating_atoms=2, radius_gyration=4.0)
    inter = dict(contact_frac=0.6, hbond_contacts=2, hydrophobic_contacts=8,
                 electrostatic_complementarity=0.3, clash=0,
                 metal_proximity_score=0.5)
    quantum = dict(delta_e=-0.4, ct_weight=0.1)
    return lig, inter, quantum


def test_feature_vector_ordering():
    lig, inter, quantum = base_features()
    vec = build_feature_vector(lig, inter, 0.45, quantum)
    assert vec.shape == (N_FEATURES,)
    fd = dict(zip(FEATURE_NAMES, vec.tolist()))
    assert fd["mw"] == 350.0
    assert fd["quantum_delta_e"] == -0.4
    assert fd["contact_frac"] == 0.6
    # order of dict matches FEATURE_NAMES exactly
    assert list(fd.keys()) == FEATURE_NAMES


def test_expert_score_bounds_and_known_boost():
    lig, inter, quantum = base_features()
    fd = dict(zip(FEATURE_NAMES, build_feature_vector(lig, inter, 0.45, quantum)))
    base = expert_score(fd, "class_a", "", 0.0, False)
    boosted = expert_score(fd, "class_a", "class_a", 0.0, False)
    mbl = expert_score(fd, "class_b", "class_b", 0.0, True)
    assert 0.0 <= base <= 1.0 and 0.0 <= boosted <= 1.0
    assert boosted > base                      # known-activity prior helps
    assert mbl > base                          # metal chelation helps on MBL
    clashy = dict(fd, clash=10)
    assert expert_score(clashy, "class_a", "", 0.0, False) < base
    assert expert_components(fd, "class_a", "class_a", 0.0, False)["known_activity_prior"] > 0


def test_expert_score_responds_to_quantum_stabilization():
    lig, inter, quantum = base_features()
    fd = dict(zip(FEATURE_NAMES, build_feature_vector(lig, inter, 0.45, quantum)))
    stabilized = expert_score(fd, "class_a", "", z_delta_e=-2.0)
    destabilized = expert_score(fd, "class_a", "", z_delta_e=+2.0)
    assert stabilized > destabilized


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_ranker_bundle_roundtrip(tmp_path):
    from amrq.ml.model import RankerBundle, build_model

    model = build_model()
    mean = np.zeros(N_FEATURES)
    std = np.ones(N_FEATURES)
    bundle = RankerBundle(model.state_dict(), mean, std, FEATURE_NAMES,
                          metrics={"train_mae": 0.1})
    path = tmp_path / "ranker.pt"
    bundle.save(path)
    loaded = RankerBundle.load(path)
    assert loaded.version == bundle.version
    assert loaded.feature_names == FEATURE_NAMES
    X = np.random.default_rng(0).normal(size=(4, N_FEATURES))
    from amrq.ml.model import predict

    scores, attributions = predict(loaded, X)
    assert scores.shape == (4,)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert len(attributions) == 4
    assert len(attributions[0]) == 5


def test_get_ranker_fallback_or_model():
    from amrq.ml.ranker import get_ranker

    ranker = get_ranker()
    assert ranker.backend in ("pytorch", "expert")
    lig, inter, quantum = base_features()
    fd = dict(zip(FEATURE_NAMES, build_feature_vector(lig, inter, 0.45, quantum)))
    X = np.array([list(fd.values())])
    scores, attrs = ranker.score_batch(
        X, [fd], "class_a", ["class_a"], np.array([0.0]), False)
    assert 0.0 <= scores[0] <= 1.0
    assert attrs
