"""Ranker facade: PyTorch model if available, expert fallback otherwise."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .expert import expert_components, expert_score
from .features import FEATURE_NAMES
from .model import MODEL_VERSION, RankerBundle, TORCH_AVAILABLE

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "ranker.pt"


class Ranker:
    """Scores feature vectors into resistance-breaking potential in [0, 1]."""

    def __init__(self, model_path: Path | str | None = None):
        self.backend = "expert"
        self.bundle: RankerBundle | None = None
        self.load_error = ""
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if TORCH_AVAILABLE and path.exists():
            try:
                self.bundle = RankerBundle.load(path)
                self.backend = "pytorch"
            except Exception as exc:  # corrupted checkpoint -> graceful fallback
                self.load_error = f"{type(exc).__name__}: {exc}"
        elif not TORCH_AVAILABLE:
            self.load_error = "torch not installed; using expert scorer"

    @property
    def info(self) -> dict:
        return {
            "backend": self.backend,
            "version": self.bundle.version if self.bundle else "expert-rules-1.0.0",
            "model_path": str(DEFAULT_MODEL_PATH),
            "load_error": self.load_error,
        }

    def score_batch(self, X: np.ndarray, feat_dicts: list[dict],
                    target_resistance_class: str, known_targets: list[str],
                    z_delta_e: np.ndarray, chelation_relevant: bool) -> tuple[np.ndarray, list]:
        """Returns (scores, attributions).

        ``feat_dicts`` are the per-row feature dicts (used by the fallback for
        per-term explanation). ``known_targets`` is the per-row list of
        resistance classes the compound is known to bind/inhibit.
        """
        X = np.asarray(X, dtype=float)
        z = np.asarray(z_delta_e, dtype=float)
        if self.backend == "pytorch":
            from .model import predict

            scores, attributions = predict(self.bundle, X)
            return scores, attributions

        scores = np.array([
            expert_score(fd, target_resistance_class, kt, zz, chelation_relevant)
            for fd, kt, zz in zip(feat_dicts, known_targets, z)
        ])
        attributions = [
            expert_components(fd, target_resistance_class, kt, zz, chelation_relevant)
            for fd, kt, zz in zip(feat_dicts, known_targets, z)
        ]
        return scores, attributions


_ranker: Ranker | None = None


def get_ranker() -> Ranker:
    """Process-wide ranker singleton (loads the trained model once)."""
    global _ranker
    if _ranker is None:
        _ranker = Ranker()
    return _ranker
