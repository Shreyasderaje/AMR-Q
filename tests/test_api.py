"""API tests (network-free): FastAPI endpoints with a stubbed pipeline."""
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amrq.api.main import app  # noqa: E402

client = TestClient(app)


def fake_results():
    """Minimal results payload with the same shape as run_pipeline output."""
    fd = {"contact_frac": 0.7, "hbond_contacts": 2, "hydrophobic_contacts": 9,
          "electrostatic_complementarity": 0.2, "clash": 0,
          "metal_proximity_score": 0.1, "chelating_atoms": 1, "mw": 300.0}
    lig = {"mw": 300.0, "crippen_logp": 1.1, "tpsa": 80.0, "hbd": 1, "hba": 4,
           "rotatable_bonds": 3, "heavy_atoms": 20, "aromatic_rings": 1,
           "fsp3": 0.2, "formal_charge": 0, "chelating_atoms": 1,
           "chelating_types": ["carboxylate"], "radius_gyration": 3.9}
    q = {"delta_e_ev": -0.31, "ct_weight": 0.05, "error": 1e-6, "energy": -9.3,
         "exact_energy": -9.3, "iterations": 120, "optimizer": "cobyla",
         "converged": True, "trace": [-9.0, -9.2, -9.3], "n_pauli_terms": 17,
         "model_params": {"eps_LH": -5.0, "eps_LL": 1.4, "eps_PH": -5.0,
                          "eps_PL": 1.2, "t_ct": 0.5, "t_ct2": 0.4, "V_total": -0.4}}
    comp = {"rank": 1, "name": "TestMol", "smiles": "CCO", "canonical_smiles": "CCO",
            "drug_class": "penicillin", "mechanism": "PBP inhibitor",
            "known_targets": ["class_a"], "inchikey": "TESTKEY",
            "feature_dict": fd, "ligand": lig, "quantum": q,
            "ml_score": 0.87, "attribution": [{"feature": "contact_frac",
                                               "contribution": 0.2}],
            "svg": "<svg xmlns='http://www.w3.org/2000/svg'></svg>"}
    return {
        "summary": {"structure": "PDB TEST", "target_key": "tem-1",
                    "target_name": "TEM-1", "elapsed_sec": 1.0,
                    "n_compounds_ranked": 1, "n_dropped": 0,
                    "backend": "statevector", "backend_used": "statevector",
                    "backends": {}, "vqe": {"mean_error_vs_exact": 1e-6,
                                            "mean_iterations": 120,
                                            "ansatz": "HF + 1x4 excitations",
                                            "optimizer": "cobyla", "qubits": 4},
                    "quantum_delta_e_stats": {"mean_ev": -0.3, "std_ev": 0.1},
                    "ranker": {"backend": "expert", "version": "test",
                               "model_path": "", "load_error": ""},
                    "versions": {"amrq": "test", "qiskit": "test", "python": "test"},
                    "seed": 42},
        "target": {"key": "tem-1", "name": "TEM-1 beta-lactamase",
                   "resistance_class": "class_a", "chelation_relevant": False,
                   "resistance_mechanism": "hydrolysis", "clinical_note": "",
                   "notes": "", "short": "TEM-1", "enzyme_class": "class A",
                   "pdb_id": "1BTL", "uniprot": "P62593", "pocket_method": "curated",
                   "pocket_chain": "A", "pocket_residues": [70]},
        "pocket": {"method": "curated", "chain": "A", "n_residues": 9,
                   "residues": ["Ser70"], "n_atoms": 60, "polar_frac": 0.4,
                   "hydrophobic_frac": 0.5, "metals": [], "missing_curated": [],
                   "centroid": [0.0, 0.0, 0.0]},
        "compounds": [comp],
        "dropped": [],
    }


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_targets_endpoint():
    r = client.get("/api/targets")
    assert r.status_code == 200
    targets = r.json()["targets"]
    assert len(targets) >= 5
    tem = next(t for t in targets if t["key"] == "tem-1")
    assert tem["pdb_id"] == "1BTL"
    assert 70 in tem["pocket_residues"]


def test_backends_endpoint():
    r = client.get("/api/backends")
    assert r.status_code == 200
    keys = {b["key"] for b in r.json()["backends"]}
    assert {"statevector", "estimator"} <= keys


def test_analyze_requires_source():
    r = client.post("/api/analyze", json={})
    assert r.status_code == 422


def test_analyze_unknown_target():
    r = client.post("/api/analyze", json={"target_key": "not-a-target"})
    assert r.status_code == 404


def test_full_job_flow_with_stubbed_pipeline(monkeypatch):
    monkeypatch.setattr("amrq.api.main.run_pipeline", lambda spec, progress=None: fake_results())
    r = client.post("/api/analyze", json={"target_key": "tem-1", "limit": 10})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    for _ in range(100):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status["status"] == "done", status

    results = client.get(f"/api/jobs/{job_id}/results").json()
    assert results["ready"] is True
    assert results["compounds"][0]["name"] == "TestMol"
    assert results["pocket"]["n_residues"] == 9

    csv_resp = client.get(f"/api/jobs/{job_id}/results.csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    lines = csv_resp.text.strip().splitlines()
    assert lines[0].startswith("rank,name")
    assert "TestMol" in lines[1]


def test_job_not_found():
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_quantum_demo_endpoint(monkeypatch):
    monkeypatch.setattr("amrq.api.main.run_quantum_demo",
                        lambda *a, **k: {"ready": True, "quantum": {"delta_e_ev": -0.2}})
    r = client.post("/api/quantum/demo",
                    json={"smiles": "CCO", "target_key": "tem-1"})
    assert r.status_code == 200
    assert r.json()["ready"] is True
