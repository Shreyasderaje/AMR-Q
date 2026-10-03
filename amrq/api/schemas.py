"""Request/response schemas for the AMR-Q API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    target_key: str | None = Field(None, description="Known target key (see /api/targets)")
    pdb_id: str | None = Field(None, description="Custom 4-letter PDB ID")
    pdb_text: str | None = Field(None, description="Raw PDB file contents")
    pocket_residues: list[int] | None = Field(None, description="Manual pocket residue numbers")
    pocket_chain: str | None = Field(None, description="Chain for manual pocket")
    limit: int = Field(40, ge=5, le=200, description="Max compounds to screen")
    backend: str = Field("statevector", description="statevector | estimator | aer | ibm")
    vqe_reps: int = Field(2, ge=1, le=4)
    optimizer: str = Field("cobyla", description="cobyla | spsa")
    seed: int = 42
    include_decoys: bool = True


class QuantumDemoRequest(BaseModel):
    smiles: str = Field(..., description="Molecule SMILES")
    target_key: str | None = None
    pdb_id: str | None = None
    pdb_text: str | None = None
    backend: str = "statevector"
    vqe_reps: int = 2
    optimizer: str = "cobyla"
    seed: int = 42
