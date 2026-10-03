#!/usr/bin/env python
"""End-to-end demo: screen a real beta-lactamase against the curated library.

Fetches TEM-1 (PDB 1BTL) from RCSB, detects the catalytic pocket, runs the
feature pipeline + 4-qubit VQE over the library, ranks with the ML model,
prints a ranked table and saves JSON + CSV outputs.

Usage:
    python scripts/run_demo.py [--target tem-1] [--limit 25] [--backend statevector]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrq.pipeline import ScreenSpec, run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="tem-1")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--backend", default="statevector")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "demo" / "demo_result.json"))
    args = parser.parse_args()

    print(f"AMR-Q demo | target={args.target} limit={args.limit} backend={args.backend}")
    print("-" * 78)
    spec = ScreenSpec(target_key=args.target, limit=args.limit, backend=args.backend)

    last = [""]
    def progress(stage, pct, msg):
        if msg != last[0]:
            last[0] = msg
            print(f"  [{pct:3.0f}%] {msg}")

    t0 = time.time()
    results = run_pipeline(spec, progress)
    elapsed = time.time() - t0

    pocket = results["pocket"]
    summary = results["summary"]
    print("-" * 78)
    print(f"structure : {summary['structure']}")
    print(f"pocket    : {pocket['n_residues']} residues ({pocket['method']}): "
          f"{', '.join(pocket['residues'][:12])}{'...' if len(pocket['residues']) > 12 else ''}")
    if pocket.get("metals"):
        print(f"metals    : {', '.join(pocket['metals'])}")
    print(f"ranker    : {summary['ranker']['backend']} ({summary['ranker']['version']})")
    print(f"vqe       : {summary['vqe']['ansatz']}; mean |E_vqe - E_exact| = "
          f"{summary['vqe']['mean_error_vs_exact']:.2e}")
    print(f"screened  : {summary['n_compounds_ranked']} compounds in {elapsed:.0f}s")
    print("-" * 78)
    hdr = f"{'#':>2}  {'compound':30s} {'score':>6} {'dE/eV':>7}  {'class':22s} known"
    print(hdr)
    print("-" * len(hdr))
    for r in results["compounds"][:15]:
        known = ";".join(r["known_targets"]) or "-"
        print(f"{r['rank']:>2}  {r['name']:30s} {r['ml_score']:6.3f} "
              f"{r['quantum']['delta_e_ev']:7.3f}  {r['drug_class'][:22]:22s} {known}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    import csv as _csv
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["rank", "name", "ml_score", "delta_e_ev", "drug_class", "smiles"])
        for r in results["compounds"]:
            w.writerow([r["rank"], r["name"], r["ml_score"],
                        r["quantum"]["delta_e_ev"], r["drug_class"], r["smiles"]])
    print(f"\nsaved: {out}")
    print(f"saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
