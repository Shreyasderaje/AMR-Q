#!/usr/bin/env python
"""Generate the weakly-supervised training set for the AMR-Q ranker.

For every (target, compound) pair the script computes the same feature
vector used at inference time, but uses *exact* diagonalization of the
4-qubit binding Hamiltonian for the quantum features (VQE converges to the
same ground state - this just skips the optimization loop, which would be
needless for label generation).

Labels come from the documented expert rule ensemble (amrq/ml/expert.py):
medicinal-chemistry priors + curated known-activity flags + the quantum
stabilization term. The result is data/training/training_set.csv, consumed
by scripts/train_ranker.py.

Usage:
    python scripts/generate_training_data.py [--limit 60] [--targets tem-1,ndm-1]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from amrq.molecules.interactions import best_pose_features  # noqa: E402
from amrq.molecules.library import load_library, validate_library  # noqa: E402
from amrq.molecules.processing import (chelating_atoms, descriptors,  # noqa: E402
                                       gasteiger_charges, prepare_molecule)
from amrq.ml.expert import expert_score  # noqa: E402
from amrq.ml.features import FEATURE_NAMES, build_feature_vector, feature_dict_from  # noqa: E402
from amrq.pipeline import ScreenSpec  # noqa: E402  (reuses constants)
from amrq.protein.fetcher import fetch_pdb  # noqa: E402
from amrq.protein.parser import detect_pocket, load_structure  # noqa: E402
from amrq.protein.pockets import load_targets  # noqa: E402
from amrq.quantum.model import build_binding_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=70,
                        help="compounds per target")
    parser.add_argument("--targets", type=str, default="",
                        help="comma-separated target keys (default: all)")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    targets = load_targets()
    keys = args.targets.split(",") if args.targets else list(targets)

    out_path = (Path(args.out) if args.out else
                ROOT / "data" / "training" / "training_set.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_library(limit=args.limit)
    df, dropped = validate_library(df)
    print(f"library: {len(df)} compounds validated ({len(dropped)} dropped)")

    # prepare molecules once (conformers are pocket-independent)
    mols = []
    for _, row in df.iterrows():
        mol = prepare_molecule(row["smiles"], seed=42)
        if mol is not None:
            mols.append((row, mol))
    print(f"prepared 3D conformers for {len(mols)} compounds")

    all_rows = []
    for key in keys:
        target = targets[key]
        t0 = time.time()
        structure = load_structure(fetch_pdb(target.pdb_id),
                                   structure_id=target.pdb_id.upper())
        pocket = detect_pocket(structure, target.pocket_method,
                               target.pocket_chain, target.pocket_residues)
        print(f"\ntarget {key}: {len(pocket.residues)} pocket residues "
              f"(method={pocket.method})")

        per_target = []
        n_failed = 0
        for row, mol in mols:
            try:
                lig_desc = descriptors(mol)
                charges = gasteiger_charges(mol)
                chel_idx, _ = chelating_atoms(mol)
                interaction, _ = best_pose_features(mol, pocket, chel_idx, charges)
            except Exception:
                n_failed += 1
                continue
            model_features = {
                **{k: interaction[k] for k in
                   ("contact_frac", "hbond_contacts", "hydrophobic_contacts",
                    "electrostatic_complementarity", "metal_proximity_score")},
                "max_negative_charge": lig_desc["max_negative_charge"],
                "max_positive_charge": lig_desc["max_positive_charge"],
                "chelating_atoms": lig_desc["chelating_atoms"],
                "pocket_polar_frac": pocket.polar_frac,
                "chelation_relevant": target.chelation_relevant,
            }
            model = build_binding_model(model_features)
            quantum = {"delta_e": model.delta_e_ev,
                       "ct_weight": model.ct_weight(model.exact_ground_state)}
            fd = feature_dict_from(lig_desc, interaction, pocket.polar_frac, quantum)
            vec = build_feature_vector(lig_desc, interaction, pocket.polar_frac, quantum)
            per_target.append({"row": row, "vec": vec, "fd": fd})

        # z-score the quantum term within the target, then label
        deltas = np.array([r["fd"]["quantum_delta_e"] for r in per_target])
        mu, sigma = deltas.mean(), deltas.std()
        for r in per_target:
            z = float((r["fd"]["quantum_delta_e"] - mu) / (sigma if sigma > 1e-9 else 1.0))
            label = expert_score(
                r["fd"], target.resistance_class,
                str(r["row"].get("known_targets", "")), z,
                target.chelation_relevant)
            all_rows.append({**dict(zip(FEATURE_NAMES, r["vec"].tolist())),
                             "label": round(label, 5),
                             "target": key,
                             "name": r["row"]["name"],
                             "resistance_class": target.resistance_class})
        print(f"  scored {len(per_target)} compounds "
              f"({n_failed} skipped) in {time.time() - t0:.0f}s")

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FEATURE_NAMES + ["label", "target",
                                                                "name", "resistance_class"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} training rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
