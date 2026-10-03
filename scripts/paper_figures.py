#!/usr/bin/env python
"""Generate publication figures for the AMR-Q paper from real pipeline outputs.

Reads outputs/demo/demo_result.json (produced by scripts/run_demo.py) and
renders vector PDF figures into paper/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "demo" / "demo_result.json"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.4,
    "figure.dpi": 150,
})

results = json.loads(DATA.read_text(encoding="utf-8"))
compounds = results["compounds"]
print(f"loaded {len(compounds)} compounds from {DATA.name}")


def convergence_figure():
    """VQE energy trace vs exact diagonalization for two representative compounds."""
    picks = []
    for name in ("Ceftazidime", "Ciprofloxacin"):
        for c in compounds:
            if c["name"].startswith(name):
                picks.append(c)
                break
    if len(picks) < 2:
        picks = compounds[:2]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    for ax, c in zip(axes, picks):
        trace = c["quantum"]["trace"]
        exact = c["quantum"]["exact_energy"]
        iters = np.arange(len(trace))
        ax.plot(iters, trace, lw=0.9, color="#0f766e", label="VQE")
        ax.axhline(exact, color="#b91c1c", lw=0.9, ls="--",
                   label=r"exact $E_0$")
        ax.set_title(c["name"], fontsize=9)
        ax.set_xlabel("COBYLA evaluation")
        if ax is axes[0]:
            ax.set_ylabel(r"$\langle H \rangle$ (model eV)")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "vqe_convergence.pdf", bbox_inches="tight")
    print("wrote vqe_convergence.pdf")


def score_distribution_figure():
    """Score separation between beta-lactams/BLIs and mechanistically unrelated compounds."""
    beta_like, other, decoy = [], [], []
    for c in compounds:
        cls = (c["drug_class"] or "").lower()
        if "decoy" in cls:
            decoy.append(c["ml_score"])
        elif any(k in cls for k in ("penicillin", "cephalosporin", "carbapenem",
                                    "monobactam", "beta-lactamase inhibitor")):
            beta_like.append(c["ml_score"])
        else:
            other.append(c["ml_score"])

    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    groups = [
        (r"$\beta$-lactams / BLIs", beta_like, "#0f766e"),
        ("other antibiotics", other, "#64748b"),
        ("decoys", decoy if decoy else [], "#b91c1c"),
    ]
    positions = np.arange(len(groups))
    for pos, (label, vals, color) in zip(positions, groups):
        if vals:
            ax.scatter(np.full(len(vals), pos) + np.random.default_rng(0)
                       .uniform(-0.08, 0.08, len(vals)), vals,
                       s=12, alpha=0.75, color=color, edgecolors="none")
        if vals:
            mean = float(np.mean(vals))
            ax.hlines(mean, pos - 0.22, pos + 0.22, color=color, lw=1.6)
    ax.set_xticks(positions, [g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel("resistance-breaking score")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "score_separation.pdf", bbox_inches="tight")
    print(f"wrote score_separation.pdf "
          f"(beta-like n={len(beta_like)}, other n={len(other)}, decoy n={len(decoy)})")


def delta_scatter_figure():
    """Classical interaction features vs quantum stabilization, colored by class family."""
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for c in compounds:
        cls = (c["drug_class"] or "").lower()
        if "decoy" in cls:
            color, marker = "#b91c1c", "x"
        elif any(k in cls for k in ("penicillin", "cephalosporin", "carbapenem",
                                    "monobactam", "beta-lactamase inhibitor")):
            color, marker = "#0f766e", "o"
        else:
            color, marker = "#64748b", "s"
        ax.scatter(-c["quantum"]["delta_e_ev"], c["ml_score"], s=13,
                   color=color, marker=marker, alpha=0.8, edgecolors="none")
    ax.set_xlabel(r"quantum stabilization $-\Delta E$ (model eV)")
    ax.set_ylabel("resistance-breaking score")
    fig.tight_layout()
    fig.savefig(OUT / "quantum_score_scatter.pdf", bbox_inches="tight")
    print("wrote quantum_score_scatter.pdf")


if __name__ == "__main__":
    convergence_figure()
    score_distribution_figure()
    delta_scatter_figure()
