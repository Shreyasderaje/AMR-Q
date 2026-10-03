# AMR-Q Research Notes

Research conducted 2026-10-03 via live verification of primary sources (APIs fetched
directly, statistics traced to their original publications). This document records what
the project's design decisions are based on, with sources. Facts are marked
**[verified]** (fetched directly) or **[cited]** (from the named publication).

---

## 1. The antimicrobial resistance (AMR) burden

| Figure | Value | Source |
|---|---|---|
| Deaths *attributable* to bacterial AMR, 2019 | **1.27 million** | Murray et al., *Lancet* 2022;399:629–655, DOI [10.1016/S0140-6736(21)02724-0](https://doi.org/10.1016/S0140-6736(21)02724-0) **[cited]** |
| Deaths *associated* with bacterial AMR, 2019 | **4.95 million** | same |
| Attributable deaths, 2021 | **1.14 million** (4.71M associated) | GBD 2021 AMR Collaborators (Naghavi et al.), *Lancet* 2024;404:1199–1226, DOI [10.1016/S0140-6736(24)01867-1](https://doi.org/10.1016/S0140-6736(24)01867-1) **[verified via PubMed 39299261]** |
| Forecast attributable deaths, 2050 | **1.91 million / yr** (8.22M associated; 39M cumulative 2025–2050) | same **[verified]** |
| "10 million deaths per year by 2050" | **O'Neill review scenario** (vs ~8.2M cancer deaths; $100T cumulative economic cost) | O'Neill, *Tackling a crisis for the health and wealth of nations* (2014) / Final report (2016), [amr-review.org](https://amr-review.org/sites/default/files/160518_Final%20paper_with%20cover.pdf) **[verified live PDF]** |
| Capitalized cost per approved drug | **~$2.6B** (2013 dollars; ~$1.4B out-of-pocket) | DiMasi, Grabowski & Hansen, *J. Health Econ.* 2016;47:20–33 **[cited]** |
| Cost per new antibiotic | **~$1.5B** | Towse/OHE 2017; DRIVE-AB final report 2018, [drive-ab.eu](https://drive-ab.eu/wp-content/uploads/2018/01/DRIVE-AB-Final-Report-Jan2018.pdf) **[cited]** |

Notes for honest communication: the widely quoted "10 million by 2050" is the O'Neill
review's scenario figure; the peer-reviewed GRAM forecast is 1.91M attributable
(8.22M associated) deaths per year by 2050 — still a ~75% increase over 2021. AMR-Q's
dashboard uses the peer-reviewed numbers with attribution.

## 2. Resistance proteins — structures and catalytic residues (verified)

All PDB entries verified live at `https://files.rcsb.org/download/{ID}.pdb` on 2026-10-03.

| Target | PDB | UniProt | Catalytic / pocket residues (Ambler numbering) |
|---|---|---|---|
| TEM-1 β-lactamase (class A) | [1BTL](https://www.rcsb.org/structure/1BTL) (1.8 Å) | P62593 | Ser70, Lys73, Ser130, Asn132, Glu166, Lys234, Thr/Ser235, Gly236 — **confirmed from PDB SITE records** |
| KPC-2 carbapenemase (class A) | [2OV5](https://www.rcsb.org/structure/2OV5) | Q9F663 | class A set (Ser70…Gly236) — **confirmed from SITE records** |
| CTX-M-14 ESBL (class A) | [1YLT](https://www.rcsb.org/structure/1YLT) (atomic resolution) | Q9L5C7 | standard class A set; file numbering starts ~residue 25 (mature protein) |
| NDM-1 metallo-β-lactamase (class B) | [4HL2](https://www.rcsb.org/structure/4HL2) (1.05 Å) | C7C422 | Zn1: **His120, His122, His189**; Zn2: **Asp124, Cys208, His250** — **confirmed** (evidence PubMed:22713171) |
| AmpC β-lactamase (class C) | [1KE4](https://www.rcsb.org/structure/1KE4) | P00811 | motifs SXXK (**Ser64, Lys67**), YXN (**Tyr150, Asn152**), KTG (**Lys315, Thr316, Gly317**), plus **Gln120** (R1-amide partner). ⚠️ Common mis-assignment: AmpC has **Gln120**, *not* Glu166/171 (those are class A residues) |
| AcrB efflux pump (RND) | [4DX5](https://www.rcsb.org/structure/4DX5) (with D13-9001, Nakashima *Nature* 2013) | P31224 | distal "hydrophobic trap": **Val605, Phe610, Ile626, Phe615, Phe617, Phe628** (Vargiu 2015, [PubMed 25114133](https://pubmed.ncbi.nlm.nih.gov/25114133/)); the 5ENO MBX2319 SITE records add **Phe178, Ala279, Pro326, Tyr327, Met573, Val612** |

Numbering caveat (matters when scripting): Ambler/PDB numbering ≠ UniProt indexing
(TEM-1 Ambler Ser70 = UniProt ACT_SITE 68; KPC-2 UniProt 69 = Ambler Ser70). AMR-Q
works in PDB numbering.

## 3. Inhibitor landscape used for the curated library

**Serine β-lactamase inhibitors (clinical):** clavulanic acid / sulbactam / tazobactam
(suicide inhibitors, class A); avibactam & relebactam (diazabicyclooctanes, class A+C);
vaborbactam (cyclic boronic acid, class A incl. KPC); durlobactam (class A+C+D).
None inhibit class B metallo-enzymes — this is exactly the gap the NDM-1 target in
AMR-Q's panel demonstrates.

**NDM-1 (metallo) inhibitors — chelation/covalent mechanisms [cited]:**
- **Aspergillomarasmine A** — natural Zn(II) chelator, rescues meropenem in vivo; King et al., *Nature* 2014;510:503–506, [PubMed 24965651](https://pubmed.ncbi.nlm.nih.gov/24965651/)
- **Ebselen** — covalent S–Se bond to Cys221 + Zn sequestration ([PubMed 41229169](https://pubmed.ncbi.nlm.nih.gov/41229169/))
- **Captopril** — binds the di-zinc active site ([PubMed 42633358](https://pubmed.ncbi.nlm.nih.gov/42633358/))
- **Dipicolinic acid** — Zn-chelating MBL-inhibitor scaffold ([review, PubMed 39203022](https://pubmed.ncbi.nlm.nih.gov/39203022/))
- **Ciclopirox** — reported in repurposing literature ([PubMed 42358646](https://pubmed.ncbi.nlm.nih.gov/42358646/)); no dedicated primary enzyme-inhibition paper found — treat as reported-only

This is why AMR-Q's quantum model scores **metal chelation explicitly** for class B
targets: it is the actual mechanism of the only known NDM-1 inhibitor chemotypes.

## 4. Quantum computing for drug discovery — where the state of the art really is

- **Field perspective:** Santagati et al., "Drug design on quantum computers," *Nature Physics* 20, 549–557 (2024), DOI [10.1038/s41567-024-02411-5](https://doi.org/10.1038/s41567-024-02411-5): ligand binding is a key use case; practical value requires fault-tolerant hardware; **reduced active spaces are the near-term path**.
- **Resource estimates:** Blunt et al., *J. Chem. Theory Comput.* (2022), [arXiv:2206.00551](https://arxiv.org/abs/2206.00551): ~50-orbital active spaces need error-corrected machines (1000+ years Trotterized → days with qubitization). One qubit per spin-orbital ⇒ a minimal-basis drug-scale molecule needs hundreds of qubits; full protein–ligand simulation is out of reach on simulators and NISQ hardware.
- **Closest published analog to AMR-Q:** Kitajima et al. (Tanabe Pharma), "Quantum Computing Calculations of Protein–Ligand Binding Energies Using Decomposition Methods on Simulated and Real Quantum Hardware," *J. Chem. Inf. Model.* (2026), DOI [10.1021/acs.jcim.6c00544](https://doi.org/10.1021/acs.jcim.6c00544): DMET decomposition of thrombin–ligand complexes; protein as ~4,000 point charges; **VQE on a 4-qubit (2e,2o) active space** on real hardware; outcome measured as *improved rank correlation* (R² = 0.762) with experimental binding enthalpies — relative ranking, not absolute energies. AMR-Q's 4-qubit model lives at exactly this scale.
- **Related protein–ligand quantum work:** Malone et al., *Phys. Chem. Chem. Phys.* 24, 15948 (2022) (VQE reduced density matrices → SAPT interaction energies); Ding & Jiang, *Int. J. Quantum Chem.* 122, e26975 (2022).
- **Reduced models are established practice:** DMET/embedding (Wouters et al., [arXiv:1605.05547](https://arxiv.org/abs/1605.05547)); DMET+VQE on the Hubbard model (Phys. Rev. B 105, 125117 (2022), [arXiv:2108.08611](https://arxiv.org/abs/2108.08611)); asymmetric Hubbard / donor–acceptor charge-transfer dimers as the minimal binding-interaction model. AMR-Q's two-site model is this construction, parameterized from docking-lite features.
- **Quantum × AMR specifically:** the intersection is small — QSVM classification of non-hemolytic antimicrobial peptides (Zhuang et al., [arXiv:2402.03847](https://arxiv.org/abs/2402.03847)); peptide binding classification on trapped-ion hardware (London et al., *npj Quantum Inf.* 2024, [arXiv:2311.15696](https://arxiv.org/abs/2311.15696)); quantum-annealing antimicrobial peptide design (*ACS Med. Chem. Lett.* 2023, DOI [10.1021/acsmedchemlett.3c00058](https://doi.org/10.1021/acsmedchemlett.3c00058)). **No flagship quantum-vs-AMR pipeline exists** — AMR-Q's positioning (quantum scoring of resistance-protein binders) is genuinely unoccupied territory, as a demonstration.

## 5. Qiskit API status (verified late 2026)

- Current stable: **Qiskit 2.5** (2.5.2, Aug 2026). V1 primitives **removed** in 2.0.
- `qiskit.primitives.StatevectorEstimator` (V2) — **valid, current**; PUB interface `(circuit, observable, params)`.
- `SparsePauliOp.from_operator()` — **valid** (Hantzko–Binkowski–Gupta 2023 decomposition).
- `Statevector.expectation_value()` — **valid**.
- `qiskit.circuit.library.TwoLocal` / `EfficientSU2` — **deprecated since 2.1** (removal in 3.0). AMR-Q therefore implements its own **particle-conserving single-excitation (Givens) ansatz** from CX/CRY primitives — hardware-ready and deprecation-proof.
- `qiskit-algorithms` — **no longer IBM-supported**; AMR-Q implements its own VQE loop (COBYLA / built-in SPSA) instead.
- AMR-Q pins `qiskit>=1.2,<3` and is tested against 2.5.2.

## 6. Open data APIs (all verified live 2026-10-03)

- **RCSB:** `https://files.rcsb.org/download/{ID}.pdb` — all 8 panel structures return 200.
- **PubChem PUG REST:** `.../compound/name/{name}/property/SMILES,ConnectivitySMILES/JSON`.
  ⚠️ Response keys were renamed (`IsomericSMILES`→`SMILES`, `CanonicalSMILES`→`ConnectivitySMILES`); `ConnectivitySMILES` is *not* salt-stripped — AMR-Q strips salts by keeping the largest RDKit fragment.
- **ChEMBL:** `https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={name}` (`.json` on the path, not the query).
- **UniProt:** `https://rest.uniprot.org/uniprotkb/{accession}.json`.

## 7. What this means for AMR-Q's claims

Established by this research and implemented honestly:
1. A 4-qubit VQE binding model is **current state-of-the-art scale** (Kitajima 2026), not a toy below it.
2. Reduced two-site/Hubbard-dimer models are **accepted reduced-model practice**.
3. Quantum contributions in the literature are evaluated as **relative ranking signals** — AMR-Q does the same (z-scored stabilization energy as one feature among 23).
4. Nothing here predicts absolute binding free energies; the README and dashboard say so explicitly.
