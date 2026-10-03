"""End-to-end pipeline integration test using the local PDB cache.

Skipped automatically when data/cache/pdb/1BTL.pdb is absent (run
scripts/run_demo.py once to populate the cache, or let it fetch online).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE_1BTL = ROOT / "data" / "cache" / "pdb" / "1BTL.pdb"

pytestmark = pytest.mark.skipif(not CACHE_1BTL.exists(),
                                reason="1BTL not in local cache yet")


def test_pipeline_end_to_end_with_cached_structure(monkeypatch):
    import numpy as np

    import amrq.pipeline as pipeline

    monkeypatch.setattr(pipeline, "fetch_pdb", lambda pdb_id, **k: CACHE_1BTL)
    spec = pipeline.ScreenSpec(target_key="tem-1", limit=8, seed=1)
    progress_events = []
    results = pipeline.run_pipeline(spec, lambda s, p, m: progress_events.append((s, p)))

    assert results["summary"]["n_compounds_ranked"] >= 6
    assert results["pocket"]["method"] in ("curated", "metal", "cocrystal")
    assert len(progress_events) > 5

    top = results["compounds"][0]
    assert 0.0 <= top["ml_score"] <= 1.0
    assert top["quantum"]["error"] < 1e-2
    assert len(top["svg"]) > 100

    deltas = np.array([c["quantum"]["delta_e_ev"] for c in results["compounds"]])
    assert (deltas <= 0.05).all()  # coupled complexes are stabilized or ~neutral

    # ranking is strictly descending
    scores = [c["ml_score"] for c in results["compounds"]]
    assert scores == sorted(scores, reverse=True)
