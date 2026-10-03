"""AMR-Q end-to-end screening pipeline.

protein structure -> binding pocket -> compound library -> interaction
features -> 4-qubit VQE binding model -> ML resistance-breaking score ->
ranked hits.
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field

import numpy as np
import qiskit

from .molecules.interactions import best_pose_features
from .molecules.library import load_library, validate_library
from .molecules.processing import (chelating_atoms, descriptors, gasteiger_charges,
                                   ligand_coords, mol_to_svg, prepare_molecule)
from .ml.features import build_feature_vector, feature_dict_from
from .ml.ranker import get_ranker
from .protein.fetcher import fetch_pdb
from .protein.parser import detect_pocket, load_structure
from .protein.pockets import get_target, load_targets
from .quantum.backends import available_backends, make_evaluator
from .quantum.model import build_binding_model
from .quantum.vqe import run_vqe, stable_seed

__version__ = "1.0.0"
TRACE_POINTS = 140  # downsampled convergence trace sent to the dashboard


@dataclass
class ScreenSpec:
    target_key: str | None = None        # known target key, or...
    pdb_id: str | None = None            # ...custom PDB ID, or...
    pdb_text: str | None = None          # ...raw uploaded PDB text
    pocket_residues: list[int] | None = None   # manual pocket override
    pocket_chain: str | None = None
    limit: int = 40
    backend: str = "statevector"
    vqe_reps: int = 2
    vqe_maxiter: int = 400
    optimizer: str = "cobyla"
    seed: int = 42
    include_decoys: bool = True


def _downsample(trace: list[float], n: int = TRACE_POINTS) -> list[float]:
    if len(trace) <= n:
        return trace
    idx = np.linspace(0, len(trace) - 1, n).astype(int)
    return [trace[i] for i in idx]


def _evaluate_compound(row: dict, mol, pocket, target_ctx: dict,
                       evaluator) -> dict | None:
    """All science for one (compound, pocket) pair. Returns a result row."""
    lig_desc = descriptors(mol)
    charges = gasteiger_charges(mol)
    chel_idx, _ = chelating_atoms(mol)
    interaction, _pose = best_pose_features(mol, pocket, chel_idx, charges)

    model_features = {
        **{k: interaction[k] for k in
           ("contact_frac", "hbond_contacts", "hydrophobic_contacts",
            "electrostatic_complementarity", "metal_proximity_score")},
        "max_negative_charge": lig_desc["max_negative_charge"],
        "max_positive_charge": lig_desc["max_positive_charge"],
        "chelating_atoms": lig_desc["chelating_atoms"],
        "pocket_polar_frac": pocket.polar_frac,
        "chelation_relevant": target_ctx["chelation_relevant"],
    }
    model = build_binding_model(model_features)
    vqe = run_vqe(model, reps=target_ctx["vqe_reps"],
                  optimizer=target_ctx["optimizer"],
                  maxiter=target_ctx["vqe_maxiter"],
                  seed=stable_seed(row.get("inchikey", row["name"]),
                                   target_ctx["key"], target_ctx["seed"]),
                  evaluator=evaluator)

    quantum = {
        "delta_e": vqe.energy - model.decoupled_energy,
        "exact_delta_e": model.delta_e_ev,
        "ct_weight": vqe.ct_weight,
        "energy": vqe.energy,
        "decoupled_energy": model.decoupled_energy,
    }
    fd = feature_dict_from(lig_desc, interaction, pocket.polar_frac, quantum)
    vec = build_feature_vector(lig_desc, interaction, pocket.polar_frac, quantum)

    return {
        "name": row["name"],
        "smiles": row["smiles"],
        "canonical_smiles": lig_desc["smiles"],
        "drug_class": row.get("drug_class", ""),
        "mechanism": row.get("mechanism", ""),
        "known_targets": [t for t in str(row.get("known_targets", "")).split(";") if t],
        "inchikey": row.get("inchikey", ""),
        "features_vector": vec.tolist(),
        "feature_dict": fd,
        "ligand": {k: lig_desc[k] for k in
                   ("mw", "crippen_logp", "tpsa", "hbd", "hba", "rotatable_bonds",
                    "heavy_atoms", "aromatic_rings", "fsp3", "formal_charge",
                    "chelating_atoms", "chelating_types", "radius_gyration")},
        "interaction": interaction,
        "quantum": {
            **vqe.to_meta(),
            "delta_e_ev": round(quantum["delta_e"], 4),
            "exact_delta_e_ev": round(quantum["exact_delta_e"], 4),
            "decoupled_energy": round(model.decoupled_energy, 4),
            "trace": _downsample(vqe.trace),
            "n_pauli_terms": model.n_pauli_terms,
            "model_params": {
                "eps_LH": round(model.eps["LH"], 3), "eps_LL": round(model.eps["LL"], 3),
                "eps_PH": round(model.eps["PH"], 3), "eps_PL": round(model.eps["PL"], 3),
                "t_ct": round(model.t_ct, 3), "t_ct2": round(model.t_ct2, 3),
                "V_total": round(sum(model.V.values()), 3),
            },
        },
        "svg": mol_to_svg(mol),
    }


def run_pipeline(spec: ScreenSpec, progress=lambda stage, pct, msg: None) -> dict:
    """Run a full screening job. ``progress(stage, pct, message)`` is called
    as the pipeline advances (used for live job status)."""
    t0 = time.time()
    progress("structure", 2, "Loading protein structure...")

    # --- resolve target + structure ---------------------------------------
    target = None
    if spec.target_key:
        target = get_target(spec.target_key)
        pdb_source = target.pdb_id
    elif spec.pdb_id:
        pdb_source = spec.pdb_id
    elif spec.pdb_text:
        pdb_source = None
    else:
        raise ValueError("Provide target_key, pdb_id or pdb_text")

    if spec.pdb_text:
        structure = load_structure(spec.pdb_text, structure_id="uploaded")
        structure_label = "uploaded PDB"
    else:
        pdb_path = fetch_pdb(pdb_source)
        structure = load_structure(pdb_path, structure_id=str(pdb_source).upper())
        structure_label = f"PDB {str(pdb_source).upper()}"

    target_ctx = {
        "key": target.key if target else "custom",
        "name": target.name if target else f"Custom target ({structure_label})",
        "resistance_class": target.resistance_class if target else "unknown",
        "chelation_relevant": target.chelation_relevant if target else False,
        "vqe_reps": spec.vqe_reps,
        "vqe_maxiter": spec.vqe_maxiter,
        "optimizer": spec.optimizer,
        "seed": spec.seed,
    }

    # --- pocket detection ---------------------------------------------------
    progress("pocket", 8, "Detecting binding pocket...")
    if spec.pocket_residues:
        chain = spec.pocket_chain or (target.pocket_chain if target else "A")
        pocket = detect_pocket(structure, "manual", chain, spec.pocket_residues)
    elif target:
        pocket = detect_pocket(structure, target.pocket_method,
                               target.pocket_chain, target.pocket_residues)
    else:
        pocket = detect_pocket(structure, "cocrystal")

    # --- library -------------------------------------------------------------
    progress("library", 12, "Loading and preparing compound library...")
    df = load_library(limit=spec.limit)
    if not spec.include_decoys:
        df = df[~df["drug_class"].str.contains("decoy", case=False, na=False)]
    df, dropped = validate_library(df)

    prepared = []
    for _, row in df.iterrows():
        mol = prepare_molecule(row["smiles"], seed=spec.seed)
        if mol is None:
            dropped.append({"name": row["name"], "reason": "3D embedding failed"})
            continue
        if mol.GetNumHeavyAtoms() > 80:
            dropped.append({"name": row["name"], "reason": "too large (>80 heavy atoms)"})
            continue
        prepared.append((row, mol))
    if not prepared:
        raise ValueError("No compounds could be prepared from the library")

    backend_infos = {b.key: {"label": b.label, "available": b.available,
                             "detail": b.detail} for b in available_backends()}
    backend_used = spec.backend if _backend_ready(spec.backend) else "statevector"
    evaluator = make_evaluator(backend_used)

    # --- main loop: features -> quantum -> per-compound ----------------------
    rows = []
    n = len(prepared)
    for i, (row, mol) in enumerate(prepared):
        pct = 15 + int(70 * i / n)
        progress("quantum", pct, f"[{i + 1}/{n}] VQE for {row['name']}")
        try:
            result = _evaluate_compound(dict(row), mol, pocket, target_ctx, evaluator)
        except Exception as exc:
            dropped.append({"name": row["name"],
                            "reason": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(result)

    if not rows:
        raise RuntimeError("Every compound failed to evaluate; nothing to rank")

    # --- ML ranking ----------------------------------------------------------
    progress("ranking", 88, "Scoring and ranking with the ML ranker...")
    X = np.array([r["features_vector"] for r in rows])
    delta_es = np.array([r["quantum"]["delta_e_ev"] for r in rows])
    mu, sigma = float(delta_es.mean()), float(delta_es.std())
    z = (delta_es - mu) / (sigma if sigma > 1e-9 else 1.0)
    for r, zz in zip(rows, z):
        r["feature_dict"]["quantum_z"] = float(zz)

    ranker = get_ranker()
    scores, attributions = ranker.score_batch(
        X,
        [r["feature_dict"] for r in rows],
        target_ctx["resistance_class"],
        [r["known_targets"] for r in rows],
        z,
        target_ctx["chelation_relevant"],
    )
    for r, score, attr in zip(rows, scores, attributions):
        r["ml_score"] = round(float(score), 4)
        r["attribution"] = attr
        r["feature_dict"].pop("quantum_z", None)

    rows.sort(key=lambda r: -r["ml_score"])
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
        r.pop("features_vector", None)   # bulky, not needed client-side

    elapsed = round(time.time() - t0, 1)
    summary = {
        "structure": structure_label,
        "target_key": target_ctx["key"],
        "target_name": target_ctx["name"],
        "elapsed_sec": elapsed,
        "n_compounds_ranked": len(rows),
        "n_dropped": len(dropped),
        "backend": spec.backend,
        "backend_used": backend_used,
        "backends": backend_infos,
        "vqe": {
            "ansatz": f"HF + {spec.vqe_reps} layer(s) of 4 single-excitation gates",
            "qubits": 4,
            "optimizer": spec.optimizer,
            "mean_error_vs_exact": round(float(np.mean(
                [r["quantum"]["error"] for r in rows])), 6),
            "mean_iterations": round(float(np.mean(
                [r["quantum"]["iterations"] for r in rows])), 1),
        },
        "quantum_delta_e_stats": {"mean_ev": round(mu, 4), "std_ev": round(sigma, 4)},
        "ranker": ranker.info,
        "versions": {"amrq": __version__, "qiskit": qiskit.__version__,
                     "python": platform.python_version()},
        "seed": spec.seed,
    }

    progress("done", 100, "Done.")
    return {
        "summary": summary,
        "target": target.summary() if target else {
            "key": "custom", "name": target_ctx["name"],
            "resistance_class": "unknown", "chelation_relevant": False,
            "resistance_mechanism": "", "clinical_note": "",
            "notes": "Custom target: pocket detected from the co-crystal ligand "
                     "or manually specified residues."},
        "pocket": pocket.summary(),
        "compounds": rows,
        "dropped": dropped,
    }


def _backend_ready(backend: str) -> bool:
    info = {b.key: b for b in available_backends()}
    return backend in info and info[backend].available


def run_quantum_demo(smiles: str, target_key: str | None = None,
                     pdb_id: str | None = None, pdb_text: str | None = None,
                     backend: str = "statevector", vqe_reps: int = 2,
                     optimizer: str = "cobyla", seed: int = 42) -> dict:
    """Run the full per-compound science (features -> VQE -> model diagnostics)
    for a single molecule - powers the dashboard's Quantum Lab panel."""
    from .molecules.processing import parse_smiles

    mol = parse_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = prepare_molecule(smiles, seed=seed)
    if mol is None:
        raise ValueError("Could not embed the molecule in 3D")

    target = get_target(target_key) if target_key else None
    if target:
        structure = load_structure(fetch_pdb(target.pdb_id), structure_id=target.pdb_id.upper())
        pocket = detect_pocket(structure, target.pocket_method,
                               target.pocket_chain, target.pocket_residues)
        label = f"PDB {target.pdb_id.upper()}"
    elif pdb_text:
        structure = load_structure(pdb_text, structure_id="uploaded")
        pocket = detect_pocket(structure, "cocrystal")
        label = "uploaded PDB"
    elif pdb_id:
        structure = load_structure(fetch_pdb(pdb_id), structure_id=pdb_id.upper())
        pocket = detect_pocket(structure, "cocrystal")
        label = f"PDB {pdb_id.upper()}"
    else:
        raise ValueError("Provide target_key, pdb_id or pdb_text")

    target_ctx = {
        "key": target.key if target else "custom",
        "name": target.name if target else label,
        "resistance_class": target.resistance_class if target else "unknown",
        "chelation_relevant": target.chelation_relevant if target else False,
        "vqe_reps": vqe_reps, "vqe_maxiter": 600, "optimizer": optimizer, "seed": seed,
    }
    backend_used = backend if _backend_ready(backend) else "statevector"
    evaluator = make_evaluator(backend_used)
    result = _evaluate_compound(
        {"name": "demo", "smiles": smiles, "inchikey": "demo"},
        mol, pocket, target_ctx, evaluator)
    result["structure"] = label
    result["pocket"] = pocket.summary()
    result["backend_used"] = backend_used
    return result
