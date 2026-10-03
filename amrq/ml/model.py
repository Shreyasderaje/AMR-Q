"""PyTorch ranking model: small MLP, sigmoid output, input-gradient attribution."""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is optional at runtime
    torch = None
    nn = None
    TORCH_AVAILABLE = False

from .features import N_FEATURES

MODEL_VERSION = "amrq-ranker-1.0.0"


def build_model(hidden=(64, 32)) -> "nn.Module":
    layers = []
    prev = N_FEATURES
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers)


class RankerBundle:
    """Model + scaler + metadata, as persisted to data/models/ranker.pt."""

    def __init__(self, state_dict, scaler_mean, scaler_std,
                 feature_names, version=MODEL_VERSION, metrics=None):
        self.scaler_mean = np.asarray(scaler_mean, dtype=float)
        self.scaler_std = np.asarray(scaler_std, dtype=float)
        self.feature_names = list(feature_names)
        self.version = version
        self.metrics = metrics or {}
        if TORCH_AVAILABLE:
            self.model = build_model()
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            self.model = None

    def to_payload(self):
        return {
            "version": self.version,
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_std": self.scaler_std.tolist(),
            "metrics": self.metrics,
            "state_dict": self.model.state_dict(),
        }

    @classmethod
    def load(cls, path) -> "RankerBundle":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            state_dict=payload["state_dict"],
            scaler_mean=payload["scaler_mean"],
            scaler_std=payload["scaler_std"],
            feature_names=payload["feature_names"],
            version=payload.get("version", MODEL_VERSION),
            metrics=payload.get("metrics", {}),
        )

    def save(self, path):
        torch.save(self.to_payload(), path)


def predict(bundle: RankerBundle, X: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Batch scores in [0,1] + per-row top-k input-gradient attributions.

    The attribution uses gradients of the output w.r.t. the *standardized*
    inputs, so contributions are comparable across features.
    """
    X = np.asarray(X, dtype=float)
    Xs = (X - bundle.scaler_mean) / np.where(bundle.scaler_std > 1e-8, bundle.scaler_std, 1.0)
    tensor = torch.tensor(Xs, dtype=torch.float32, requires_grad=True)
    with torch.enable_grad():
        out = bundle.model(tensor).squeeze(-1)
        out.sum().backward()
    grads = tensor.grad.detach().numpy()
    scores = np.clip(out.detach().numpy(), 0.0, 1.0)
    attributions = []
    for i, (row_scores, row_grads) in enumerate(zip(scores, grads)):
        contrib = row_grads * Xs[i]
        order = np.argsort(-np.abs(contrib))[:5]
        attributions.append([
            {"feature": bundle.feature_names[j], "contribution": float(contrib[j])}
            for j in order
        ])
    return scores, attributions
