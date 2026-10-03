# AMR-Q — Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          React dashboard (frontend/)                    │
│        console · live progress · ranked table · quantum reports        │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ fetch /api/*  (JSON, poll-based jobs)
┌──────────────────────────────▼─────────────────────────────────────────┐
│                        FastAPI server (amrq/api/)                       │
│   /api/targets  /api/backends  /api/analyze  /api/jobs  /api/quantum   │
│        background job runner · in-memory + on-disk job store           │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────────┐
│                       Pipeline (amrq/pipeline.py)                       │
│                                                                       │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────────────┐  │
│  │ protein/     │   │ molecules/   │   │ quantum/                  │  │
│  │  fetcher     │   │  library     │   │  fermion.py (JW algebra)  │  │
│  │  parser      │──▶│  processing  │──▶│  model.py (2-site H)      │  │
│  │  pockets     │   │  interactions│   │  vqe.py (COBYLA/SPSA)     │  │
│  └──────────────┘   └──────────────┘   │  backends.py (4 backends) │  │
│         │                    │         └────────────┬──────────────┘  │
│         ▼                    ▼                      ▼                 │
│  RCSB PDB cache      RDKit (ETKDG,          SparsePauliOp · 4 qubits  │
│  curated targets     MMFF, Gasteiger,       exact-diag benchmark      │
│  (JSON, extensible)  KD-tree contacts)      per compound              │
│                                              │                        │
│  ┌───────────────────────────────────────────▼──────────────────────┐ │
│  │ ml/  features.py (23 features) · expert.py (weak supervision)    │ │
│  │      model.py (MLP + attribution) · train.py · ranker.py         │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

## Module map

| Path | Responsibility |
|---|---|
| `amrq/quantum/fermion.py` | creation/annihilation/number operators as dense matrices over the Jordan-Wigner basis (Qiskit little-endian convention) |
| `amrq/quantum/model.py` | builds the 16×16 two-site binding Hamiltonian from interaction features; Pauli decomposition; exact N=2-sector diagonalization; charge-transfer observables |
| `amrq/quantum/vqe.py` | particle-conserving excitation ansatz (CX·CRY·CX Givens gates), COBYLA + built-in SPSA optimizers, convergence tracing, deterministic seeding |
| `amrq/quantum/backends.py` | one `evaluate(qc, params, observable)` interface for statevector / EstimatorV2 / Aer / IBM runtime; graceful availability reporting |
| `amrq/protein/fetcher.py` | RCSB download with local cache (`data/cache/pdb/`) |
| `amrq/protein/parser.py` | BioPython parsing; pocket detection via curated residues, metal sites, or co-crystal ligand; `Pocket` geometry (axes, polar fractions) |
| `amrq/protein/pockets.py` | loads `data/proteins/known_targets.json` — the extensible target panel |
| `amrq/molecules/library.py` | loads `data/compounds/antibiotic_library.csv`; RDKit validation, salt handling, InChIKey dedup |
| `amrq/molecules/processing.py` | ETKDG embedding + MMFF minimization, descriptors, Gasteiger charges, metal-chelation SMARTS, 2D SVG depictions |
| `amrq/molecules/interactions.py` | pose generation (centroid + PCA alignments), KD-tree contact features, best-pose selection |
| `amrq/ml/features.py` | the canonical 23-feature ordering shared by training and inference |
| `amrq/ml/expert.py` | transparent expert-rule scorer (weak supervision; also the no-torch fallback) |
| `amrq/ml/model.py` | PyTorch MLP, checkpoint bundle (weights + scaler + metadata), gradient attribution |
| `amrq/ml/train.py` | training loop with early stopping; writes `data/models/ranker.pt` |
| `amrq/pipeline.py` | orchestration + progress reporting + summary statistics |
| `amrq/api/main.py` | FastAPI routes, job runner (threads + lock), CSV export, static frontend mount |

## Data files (all committed, all regenerable)

| File | Purpose | Regenerate |
|---|---|---|
| `data/compounds/library_source.json` | curated compound metadata (168 entries) with PubChem/ChEMBL provenance | hand-edit |
| `data/compounds/antibiotic_library.csv` | validated SMILES + InChIKeys | `python scripts/build_library.py` |
| `data/proteins/known_targets.json` | target panel (add a JSON entry = new target in the UI) | hand-edit |
| `data/training/training_set.csv` | 6 targets × 60 compounds feature/label rows | `python scripts/generate_training_data.py` |
| `data/models/ranker.pt` | trained MLP + scaler + metrics | `python scripts/train_ranker.py` |
| `data/cache/pdb/` | downloaded structures (git-ignored) | fetched on demand |

## Key design decisions

1. **Jobs, not request/response.** A screen takes ~1 s per compound; the API returns a
   `job_id` immediately and runs the pipeline in a background thread with progress
   callbacks; the dashboard polls `/api/jobs/{id}`. Results are also persisted to
   `outputs/jobs/{id}.json` (best-effort) so they survive restarts.
2. **One backend interface.** The VQE loop never knows which simulator serves it —
   statevector (default), EstimatorV2, Aer, or IBM hardware are interchangeable.
3. **Graceful degradation everywhere.** No torch → expert scorer. No qiskit-aer →
   backend marked unavailable in UI. No network → cached PDBs and committed CSV work
   offline. No trained model file → expert rules.
4. **Determinism.** Seeds for embedding, poses, ansatz init, split; VQE validated
   against exact diagonalization inside every run.
5. **Extensibility by data.** New resistance target = one JSON entry; new compound =
   one CSV row; new quantum backend = one function implementing `evaluate`.
