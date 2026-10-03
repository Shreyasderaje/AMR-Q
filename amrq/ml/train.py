"""Train the ML ranker from the weakly-supervised dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES
from .model import MODEL_VERSION, RankerBundle, TORCH_AVAILABLE, build_model


def train_from_csv(csv_path: Path | str, out_path: Path | str,
                   epochs: int = 300, lr: float = 3e-3, seed: int = 42,
                   val_frac: float = 0.15) -> dict:
    """Train the ranker MLP; returns metrics and writes the model bundle.

    The CSV needs the FEATURE_NAMES columns plus a ``label`` column
    (continuous in [0,1]) - produced by scripts/generate_training_data.py.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to train. "
                           "pip install torch (CPU: --index-url https://download.pytorch.org/whl/cpu)")

    import torch
    from torch import nn

    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_NAMES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"training CSV missing columns: {missing}")

    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(val_frac * len(X)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    mean, std = X[train_idx].mean(axis=0), X[train_idx].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    def norm(a):
        return torch.tensor((a - mean) / std, dtype=torch.float32)

    Xtr, ytr = norm(X[train_idx]), torch.tensor(y[train_idx], dtype=torch.float32)
    Xva, yva = norm(X[val_idx]), torch.tensor(y[val_idx], dtype=torch.float32)

    torch.manual_seed(seed)
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val, best_state, patience, bad = np.inf, None, 40, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr).squeeze(-1), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva).squeeze(-1), yva))
        if val_loss < best_val - 1e-6:
            best_val, bad = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_tr = model(Xtr).squeeze(-1).numpy()
        pred_va = model(Xva).squeeze(-1).numpy()
    metrics = {
        "train_mae": float(np.abs(pred_tr - ytr.numpy()).mean()),
        "val_mae": float(np.abs(pred_va - yva.numpy()).mean()),
        "val_pearson": float(np.corrcoef(pred_va, yva.numpy())[0, 1]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "epochs_run": epoch + 1,
    }

    bundle = RankerBundle(
        state_dict=model.state_dict(), scaler_mean=mean, scaler_std=std,
        feature_names=FEATURE_NAMES, version=MODEL_VERSION, metrics=metrics)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.save(out_path)
    return metrics
