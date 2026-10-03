#!/usr/bin/env python
"""Train the PyTorch ranker from the weakly-supervised dataset.

Reads data/training/training_set.csv (regenerate with
scripts/generate_training_data.py) and writes data/models/ranker.pt.

Usage:
    python scripts/train_ranker.py [--epochs 300]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrq.ml.train import train_from_csv  # noqa: E402

DEFAULT_CSV = ROOT / "data" / "training" / "training_set.csv"
DEFAULT_OUT = ROOT / "data" / "models" / "ranker.pt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    metrics = train_from_csv(args.csv, args.out, epochs=args.epochs)
    Path(args.out).with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print("metrics:", json.dumps(metrics, indent=2))
    print(f"model saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
