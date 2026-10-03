"""AMR-Q FastAPI server.

Serves the JSON API under /api and the React dashboard from ../frontend.
Run:  uvicorn amrq.api.main:app --reload  (then open http://127.0.0.1:8000)
"""
from __future__ import annotations

import csv
import io
import json
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..pipeline import ScreenSpec, run_pipeline, run_quantum_demo
from ..protein.pockets import load_targets
from ..quantum.backends import available_backends
from .schemas import AnalyzeRequest, QuantumDemoRequest

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
_JOBS_DIR = OUTPUT_DIR / "jobs"

app = FastAPI(title="AMR-Q API", version="1.0.0", description=(
    "Quantum-assisted antibiotic resistance fighter: protein -> pocket -> "
    "compounds -> VQE binding model -> ML-ranked resistance-breaking candidates."))

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _update_job(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id].update(fields)


def _progress_for(job_id: str):
    def cb(stage: str, pct: float, message: str) -> None:
        _update_job(job_id, stage=stage, progress=pct, message=message)
    return cb


def _run_job(job_id: str, spec: ScreenSpec) -> None:
    try:
        results = run_pipeline(spec, _progress_for(job_id))
        _update_job(job_id, status="done", results=results, message="Done.")
        _persist_job(job_id)
    except Exception as exc:
        _update_job(job_id, status="error", error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(), message="Job failed.")


def _persist_job(job_id: str) -> None:
    """Best-effort on-disk copy of results (survives server restarts)."""
    try:
        _JOBS_DIR.mkdir(parents=True, exist_ok=True)
        job = _JOBS.get(job_id, {})
        if job.get("results"):
            (_JOBS_DIR / f"{job_id}.json").write_text(
                json.dumps(job["results"], ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _get_job(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job {job_id}")
    return job


# --------------------------------------------------------------------------
# API routes
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AMR-Q", "version": "1.0.0"}


@app.get("/api/targets")
def targets():
    return {"targets": [t.summary() for t in load_targets().values()]}


@app.get("/api/backends")
def backends():
    return {"backends": [
        {"key": b.key, "label": b.label, "available": b.available, "detail": b.detail}
        for b in available_backends()]}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, background: BackgroundTasks):
    if not (req.target_key or req.pdb_id or req.pdb_text):
        raise HTTPException(422, "One of target_key, pdb_id or pdb_text is required")
    if req.target_key and req.target_key not in load_targets():
        raise HTTPException(404, f"Unknown target_key {req.target_key!r}")
    spec = ScreenSpec(
        target_key=req.target_key, pdb_id=req.pdb_id, pdb_text=req.pdb_text,
        pocket_residues=req.pocket_residues, pocket_chain=req.pocket_chain,
        limit=req.limit, backend=req.backend, vqe_reps=req.vqe_reps,
        optimizer=req.optimizer, seed=req.seed, include_decoys=req.include_decoys)
    return _start_job(spec, background)


@app.post("/api/analyze/upload")
async def analyze_upload(background: BackgroundTasks,
                         pdb_file: UploadFile = File(...),
                         limit: int = Form(40),
                         backend: str = Form("statevector"),
                         pocket_residues: str = Form(""),
                         pocket_chain: str = Form("A")):
    """Multipart variant: upload a .pdb file, optionally with manual pocket
    residues (comma-separated numbering as in the file)."""
    text = (await pdb_file.read()).decode("utf-8", errors="replace")
    residues = [int(r) for r in pocket_residues.replace(";", ",").split(",") if r.strip()]
    spec = ScreenSpec(
        pdb_text=text, limit=limit, backend=backend,
        pocket_residues=residues or None,
        pocket_chain=pocket_chain or None)
    return _start_job(spec, background)


def _start_job(spec: ScreenSpec, background: BackgroundTasks):
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {"id": job_id, "status": "running", "stage": "queued",
                         "progress": 0, "message": "Queued.", "created_at": _now()}
    background.add_task(_run_job, job_id, spec)
    return {"job_id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _get_job(job_id)
    return {k: job.get(k) for k in
            ("id", "status", "stage", "progress", "message", "error", "created_at")}


@app.get("/api/jobs/{job_id}/results")
def job_results(job_id: str):
    job = _get_job(job_id)
    if job.get("status") == "error":
        raise HTTPException(500, job.get("error", "job failed"))
    if "results" not in job:
        return {"ready": False, "status": job.get("status"),
                "progress": job.get("progress"), "message": job.get("message")}
    return {"ready": True, **job["results"]}


@app.get("/api/jobs/{job_id}/results.csv")
def job_results_csv(job_id: str):
    job = _get_job(job_id)
    if "results" not in job:
        raise HTTPException(404, "Results not ready")
    rows = job["results"]["compounds"]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["rank", "name", "drug_class", "ml_score",
                     "quantum_delta_e_ev", "charge_transfer_weight",
                     "vqe_error_vs_exact", "contact_frac", "hbond_contacts",
                     "hydrophobic_contacts", "electrostatic_complementarity",
                     "clash", "metal_proximity_score", "chelating_atoms",
                     "mw", "logp", "tpsa", "known_targets", "mechanism", "smiles"])
    for r in rows:
        f, l = r["feature_dict"], r["ligand"]
        writer.writerow([
            r["rank"], r["name"], r["drug_class"], r["ml_score"],
            r["quantum"]["delta_e_ev"], r["quantum"]["ct_weight"],
            r["quantum"]["error"], f["contact_frac"], f["hbond_contacts"],
            f["hydrophobic_contacts"], f["electrostatic_complementarity"],
            f["clash"], f["metal_proximity_score"], l["chelating_atoms"],
            l["mw"], l["crippen_logp"], l["tpsa"],
            ";".join(r["known_targets"]), r["mechanism"], r["smiles"]])
    buffer.seek(0)
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition":
                                      f'attachment; filename="amrq_results_{job_id}.csv"'})


@app.post("/api/quantum/demo")
def quantum_demo(req: QuantumDemoRequest):
    try:
        return {"ready": True, **run_quantum_demo(
            req.smiles, req.target_key, req.pdb_id, req.pdb_text,
            req.backend, req.vqe_reps, req.optimizer, req.seed)}
    except Exception as exc:
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")


# Serve the React dashboard last so /api routes take precedence
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
