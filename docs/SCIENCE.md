# AMR-Q — Scientific Methodology

This document explains exactly what the pipeline computes, why each choice was made,
and — just as importantly — what it does **not** do. AMR-Q is an open research
prototype for *education and early-stage prioritization*, not a validated
drug-discovery engine.

---

## 1. Pipeline overview

```
protein structure (RCSB / upload)
   └── binding pocket      curated catalytic residues | metal site | co-crystal ligand
compound library (168 curated molecules)
   └── 3D conformers       RDKit ETKDG + MMFF
   └── rigid poses         centroid match + principal-axis alignment (5 poses)
   └── interaction feats   KD-tree contacts, H-bonds, charges, chelation, clashes
   └── QUANTUM MODEL       4-qubit two-site binding Hamiltonian → VQE ground state
   └── ML ranker           PyTorch MLP over 23 classical + quantum features
ranked resistance-breaking candidates
```

## 2. The pocket

A "pocket" is a set of residues plus their heavy atoms. Three detection strategies:

1. **Curated** — known catalytic residues per target (PDB numbering; sources in
   `docs/RESEARCH.md §2`). Used for TEM-1, KPC-2, CTX-M-14, AmpC, AcrB.
2. **Metal** — residues within 6.5 Å of bound metal ions (NDM-1's di-zinc site).
   Used when the target declares `pocket.method = "metal"`; falls back to co-crystal
   detection if no metal is present.
3. **Co-crystal** — residues within 7 Å of the largest co-crystallized ligand
   (crystallization additives excluded by name). Default for custom PDB entries.

## 3. Interaction features (docking-lite)

For each compound: protonate → ETKDG embed (3 conformers) → MMFF minimize → keep the
lowest-energy conformer. The conformer is rigidly placed into the pocket in 5 candidate
orientations (pure centroid translation + 4 principal-axis alignments with sign
variants); interaction features are computed per pose and the best pose wins a simple
pre-score.

Features (per pose, computed with `scipy.cKDTree` over pocket heavy atoms):

| Feature | Definition |
|---|---|
| `contact_frac` | fraction of pocket residues with any heavy atom within 4.5 Å |
| `hbond_contacts` | ligand N/O/S within [2.4, 3.5] Å of pocket N/O |
| `hydrophobic_contacts` | ligand C within 4.5 Å of pocket C/S |
| `electrostatic_complementarity` | tanh(−Σ q_lig·q_pock/d) over pairs < 6 Å (Gasteiger ligand charges; coarse per-element pocket charges) |
| `clash` | ligand–pocket heavy-atom pairs closer than 2.6 Å |
| `metal_proximity_score` | exp(−(d_min − 2.3)²/1.8) where d_min is the closest chelating-atom-to-metal distance |

This is deliberately **not** a docking program: no flexible fitting, no solvation, no
force-field scoring. It is a transparent feature extractor for the layers above.
Known limitations follow directly: pose quality is approximate, and all features are
computed for a single static conformation.

## 4. The quantum layer

### 4.1 Why a reduced model

Full drug–protein electronic structure needs one qubit per spin-orbital — thousands of
qubits even for a minimal-basis drug — far beyond simulators and hardware (Blunt 2022;
Santagati 2024). The field's practical answer is **reduction**: active spaces, fragment
embedding (DMET), and two-site donor–acceptor models. The closest published
protein–ligand VQE demonstration (Kitajima 2026, *JCIM*) likewise ran VQE on a
**4-qubit** active space with the protein as point charges, and reported *rank
correlation* with experiment, not absolute energies. AMR-Q operates at the same scale
and with the same kind of claim.

### 4.2 The two-site binding Hamiltonian

Four spin-orbitals, two electrons (spinless), one site = ligand, one = pocket:

```
site L (ligand):  LH = donor (HOMO-like)    LL = acceptor (LUMO-like)
site P (pocket):  PH = donor (HOMO-like)    PL = acceptor (LUMO-like)

H = Σᵢ εᵢ nᵢ                                  on-site orbital energies
  + t_pol · (a†_LH a_LL + a†_PH a_PL + h.c.)  intra-site polarization
  + t_ct  · (a†_LL a_PH + h.c.)               pocket → ligand back-donation
  + t_ct2 · (a†_LH a_PL + h.c.)               ligand → pocket charge transfer
  + Σ_{i∈L, j∈P} V_ij nᵢ nⱼ                   inter-site contact stabilization
```

The Hartree-Fock reference |LH, PH⟩ (one electron per site) is basis state |0101⟩.
All hopping conserves particle number, so the ground state lives in the N=2 sector and
a particle-conserving ansatz can reach it exactly.

### 4.3 Feature → parameter mapping (eV-like units)

| Parameter | Mapping | Rationale |
|---|---|---|
| ε_LH | −5.2 + 1.6·q_neg | electron-rich ligands (large \|Gasteiger min charge\|) raise their donor level |
| ε_LL | max(1.0, 2.2 − 2.5·q_pos) | electron-poor ligands lower their acceptor level (kept > 0 so N=2 is the physical ground sector) |
| ε_PH, ε_PL | −5.0 + 0.6·polar, max(1.0, 1.9 − 1.6·polar) | polar pockets are better acceptors |
| t_ct | 0.9 · contact_frac · (0.4 + 0.6·electro) | coupling grows with contact and charge complementarity |
| t_ct2 | 0.7 · contact_frac · (0.4 + 0.6·polar) | ligand→pocket transfer favors polar pockets |
| V_ij | −(0.35/0.20/0.20/0.25)·V, V = 0.30·hb + 0.25·hyd + 0.25·elec (+ 0.35·metal + 0.10·n_chel for metallo targets) | contact stabilization distributed over the four cross-site pairs; **metal chelation enters here** because Zn²⁺ coordination is the real mechanism of NDM-1 inhibitors (RESEARCH.md §3) |
| t_pol | 0.18 (fixed) | generic polarization |

The 16×16 matrix is decomposed to a `SparsePauliOp` (Qiskit) — typically 15–25 Pauli
terms on 4 qubits.

### 4.4 VQE

- **Ansatz**: HF reference + R layers of single-excitation (Givens) gates on pairs
  (0,1), (2,3), (0,3), (1,2). Each gate is `CX(i,j) · CRY(θ; j→i) · CX(i,j)` =
  exp(−iθ/2 (XᵢYⱼ − YᵢXⱼ)) — an exact SO(2) rotation in the {|10⟩, |01⟩} subspace.
  Particle number is conserved **by construction**, so the circuit is hardware-ready
  (no penalty terms, no non-physical states) and avoids Qiskit's deprecated
  `TwoLocal`/`EfficientSU2` classes. Default: 2 layers → 8 parameters.
- **Optimizers**: COBYLA (scipy, gradient-free, deterministic) or SPSA (built-in —
  the quantum-native choice designed for noisy hardware, Spall 1992).
- **Backends** (all implementing the same interface):
  - `statevector` — exact dense simulation via `qiskit.quantum_info.Statevector`
    (the free IBM Quantum *simulator* path; default).
  - `estimator` — `qiskit.primitives.StatevectorEstimator` (V2 primitives, the same
    interface used for real hardware).
  - `aer` — shot-based/noisy simulation (optional `qiskit-aer` package).
  - `ibm` — real IBM Quantum hardware via `qiskit-ibm-runtime` (free token).
- **Verification**: every result also stores the exact N=2-sector diagonalization; the
  dashboard shows |E_VQE − E_exact| per compound. In testing VQE converges to the
  exact ground state (error < 10⁻³ eV; typically < 10⁻⁶).

### 4.5 Quantum observables used downstream

- **ΔE = E_coupled − E_decoupled** (eV-like; negative = binding-stabilizing), where the
  decoupled reference sets the inter-site couplings to zero. Z-scored across the run
  before entering the scorer so it is comparable across targets.
- **Charge-transfer weight**: ground-state probability on configurations where both
  electrons sit on the same site — a proxy for donor–acceptor interaction character.

## 5. The ML ranker

**Input**: 23 features — 14 ligand physicochemical (RDKit: MW, cLogP, TPSA, HBD/HBA,
rotatable bonds, heavy atoms, aromatic rings, Fsp³, formal charge, Gasteiger extremes,
chelating-atom count, radius of gyration), 7 interaction features (§3), 2 quantum
features (§4.5).

**Output**: resistance-breaking score ∈ [0, 1].

**Model**: PyTorch MLP 23 → 64 → 32 → 1, standardized inputs, MSE on soft labels,
Adam with early stopping. A gradient-attribution (input × grad) top-5 is computed per
compound and shown in the dashboard.

**Labels — stated plainly**: there is no experimental binding dataset behind this
project. Training labels come from a **documented expert rule ensemble**
(`amrq/ml/expert.py`): contact/H-bond/hydrophobic coverage, charge complementarity,
clash penalties, metal-chelation terms for metallo targets, the quantum stabilization
z-score, plus a +0.30 prior when a compound's curated `known_targets` metadata includes
the target's resistance class. The MLP therefore *distills these priors into a smooth
function of raw features* — useful for calibration, ranking stability and attribution,
but it is weak supervision, not learned experimental affinity. If torch is not
installed, the same expert ensemble is used directly (the API reports
`ranker.backend = "expert"`).

**Decoys**: 14 pharmacologically unrelated molecules (NSAIDs, beta-blockers, etc.) are
included with no known-target prior; they score low and serve as internal negative
controls.

## 6. What AMR-Q does NOT claim

1. **No absolute binding energies.** The quantum ΔE is a relative scoring signal in
   model-eV, z-scored per run. It is not a free energy, not solvated, not entropic.
2. **No docking accuracy.** Poses are rigid, placements are geometric, and the pocket
   is a residue list — not a cavity search.
3. **No experimental validation.** The expert-score labels encode literature-grounded
   priors, not measured affinities. A ranked list is a *hypothesis list*.
4. **No medical or clinical use.** Outputs must not inform therapy.
5. **Not medical advice.** For research and education only.

## 7. Reproducibility

- Fixed seeds everywhere: conformer embedding (`randomSeed=42`), pose generation,
  ansatz initialization (`stable_seed(inchikey, target, run_seed)`), train/val split.
- VQE is benchmarked against exact diagonalization inside every run
  (`summary.vqe.mean_error_vs_exact`).
- `scripts/run_demo.py` reproduces the demo end-to-end from a fresh clone.

## 8. Roadmap (natural next steps)

- Replace docking-lite with a real open-source docking backend (e.g., AutoDock Vina)
  feeding the same quantum + ML layers.
- Calibrate the two-site parameters against DFT interaction energies on small
  ligand–residue pairs (fitting, not hand-tuning).
- Swap the weak-supervision labels for experimental data (e.g., the BLIP
  beta-lactamase enzyme-inhibition dataset) and retrain the ranker for real.
- Run the identical ansatz on IBM Quantum hardware via the `ibm` backend and quantify
  shot-noise impact on ranking.
- Extend the target panel (OXA-48 class D, TetA efflux antiporter, vancomycin-resistance
  ligases) — adding a target is one JSON entry.
