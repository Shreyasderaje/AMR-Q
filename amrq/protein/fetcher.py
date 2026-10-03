"""Download and cache protein structures from the RCSB Protein Data Bank."""
from __future__ import annotations

import re
from pathlib import Path

import requests

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "pdb"

_PDB_ID_RE = re.compile(r"^[0-9A-Za-z]{4}$")


def validate_pdb_id(pdb_id: str) -> str:
    pid = (pdb_id or "").strip().upper()
    if not _PDB_ID_RE.match(pid):
        raise ValueError(f"Invalid PDB ID: {pdb_id!r} (expected 4 alphanumeric characters)")
    return pid


def fetch_pdb(pdb_id: str, cache_dir: Path | None = None, timeout: int = 30) -> Path:
    """Download ``{pdb_id}.pdb`` from RCSB, with local caching."""
    pid = validate_pdb_id(pdb_id)
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{pid}.pdb"
    if target.exists() and target.stat().st_size > 0:
        return target
    resp = requests.get(RCSB_URL.format(pdb_id=pid), timeout=timeout,
                        headers={"User-Agent": "AMR-Q/1.0 (open-source research tool)"})
    resp.raise_for_status()
    if "ATOM" not in resp.text:
        raise ValueError(f"RCSB returned no ATOM records for {pid}")
    target.write_text(resp.text, encoding="utf-8")
    return target
