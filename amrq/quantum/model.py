"""AMR-Q reduced two-site binding Hamiltonian.

Simulating the full electronic structure of a drug-protein complex requires
thousands of qubits (one per spin-orbital), far beyond simulators and
hardware alike. AMR-Q therefore builds a *reduced two-site model* of the
binding interface -- the same spirit as donor-acceptor / charge-transfer
(CT) dimer models and extended Hubbard models used in the quantum chemistry
literature -- and runs VQE on it. The quantum-computed ground-state
stabilization energy is used as a scoring feature for the downstream ML
ranker.

Model: 4 spin-orbitals, 2 electrons (spinless), one "site" = ligand, the
other = binding pocket.

    site L (ligand)  : orbital LH (donor / HOMO-like), LL (acceptor / LUMO-like)
    site P (pocket)  : orbital PH (donor / HOMO-like), PL (acceptor / LUMO-like)

    H = sum_i eps_i n_i                         on-site orbital energies
      + t_pol * sum_site (a^d_H a_L + h.c.)     intra-site polarization
      + t_ct  * (a^d_LL a_PH + h.c.)            pocket -> ligand back-donation
      + t_ct2 * (a^d_LH a_PL + h.c.)            ligand -> pocket charge transfer
      + sum_{i in L, j in P} V_ij n_i n_j       inter-site contact stabilization

The Hartree-Fock reference |LH, PH> (one electron per site) is the qiskit
basis state with qubits 0 and 2 occupied. Hopping terms conserve particle
number, so the ground state lives in the N=2 sector and a particle-conserving
ansatz can reach it.

Parameters are mapped from classical ligand-pocket interaction features
(Gasteiger charges, contact counts, H-bonds, metal proximity, ...); the full
mapping is documented in docs/SCIENCE.md. Energies are in "model eV": they
are calibrated to electronic scales and used *relatively* (z-scored within a
screening run), not as physical binding free energies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fermion import creation, annihilation, number, occupation_indices_with, total_number

N_ORBITALS = 4
LH, LL, PH, PL = 0, 1, 2, 3  # orbital indices
HF_STATE_INDEX = (1 << LH) | (1 << PH)  # |0101> little-endian, index 5
CT_STATE_INDICES = ((1 << LH) | (1 << LL), (1 << PH) | (1 << PL))  # both electrons on one site

# caps that keep the N=2 sector the physical ground state
_MAX_TOTAL_V = 1.2
_MIN_LUMO_EPS = 1.0


@dataclass
class BindingModel:
    """Built binding Hamiltonian + diagnostics for one (ligand, pocket) pair."""

    eps: dict[str, float]
    t_pol: float
    t_ct: float
    t_ct2: float
    V: dict[tuple[int, int], float]
    matrix: np.ndarray                       # 16x16 second-quantized Hamiltonian
    pauli_op: object                         # qiskit SparsePauliOp (4 qubits)
    hf_index: int = HF_STATE_INDEX
    exact_energy: float = 0.0                # ground energy in the N=2 sector
    exact_ground_state: np.ndarray = field(default_factory=lambda: np.empty(0))
    decoupled_energy: float = 0.0            # eps_LH + eps_PH (no coupling)
    particle_number_ok: bool = True

    @property
    def delta_e_ev(self) -> float:
        """E_coupled - E_decoupled (<= 0 means binding-stabilizing)."""
        return self.exact_energy - self.decoupled_energy

    @property
    def n_pauli_terms(self) -> int:
        return len(self.pauli_op)

    def ct_weight(self, state: np.ndarray) -> float:
        """Probability weight on charge-transfer configurations (both electrons
        on the same site) for a statevector in the full 16-dim space."""
        probs = np.abs(np.asarray(state).reshape(-1)) ** 2
        return float(sum(probs[i] for i in CT_STATE_INDICES))


def _hop(n: int, i: int, j: int) -> np.ndarray:
    """a^dagger_j a_i + a^dagger_i a_j."""
    cdag_j, a_i = creation(n, j), annihilation(n, i)
    cdag_i, a_j = creation(n, i), annihilation(n, j)
    return cdag_j @ a_i + cdag_i @ a_j


def build_binding_model(features: dict) -> BindingModel:
    """Construct the 4-qubit binding Hamiltonian from interaction features.

    ``features`` keys used: max_negative_charge, max_positive_charge,
    contact_frac, hbond_contacts, hydrophobic_contacts,
    electrostatic_complementarity, metal_proximity_score, chelating_atoms,
    pocket_polar_frac, chelation_relevant.
    """
    q_neg = abs(float(np.nan_to_num(features.get("max_negative_charge", 0.25))))
    q_pos = abs(float(np.nan_to_num(features.get("max_positive_charge", 0.2))))
    contact = float(np.clip(np.nan_to_num(features.get("contact_frac", 0.0)), 0.0, 1.0))
    hbond = float(np.nan_to_num(features.get("hbond_contacts", 0.0)))
    hydro = float(np.nan_to_num(features.get("hydrophobic_contacts", 0.0)))
    electro = float(np.clip(np.nan_to_num(features.get("electrostatic_complementarity", 0.0)), 0.0, 1.0))
    metal = float(np.clip(np.nan_to_num(features.get("metal_proximity_score", 0.0)), 0.0, 1.0))
    n_chel = float(np.nan_to_num(features.get("chelating_atoms", 0.0)))
    polar = float(np.clip(np.nan_to_num(features.get("pocket_polar_frac", 0.4)), 0.0, 1.0))
    chelation_relevant = bool(features.get("chelation_relevant", False))

    # --- on-site energies (eV-like) --------------------------------------
    eps = {
        "LH": -5.2 + 1.6 * q_neg,                    # electron-rich ligand -> higher HOMO
        "LL": max(_MIN_LUMO_EPS, 2.2 - 2.5 * q_pos), # electron-poor ligand -> deeper LUMO
        "PH": -5.0 + 0.6 * polar,
        "PL": max(_MIN_LUMO_EPS, 1.9 - 1.6 * polar), # polar pocket -> better acceptor
    }

    # --- hoppings ---------------------------------------------------------
    align = 0.4 + 0.6 * electro
    t_pol = 0.18
    t_ct = 0.9 * contact * align                       # LL <-> PH  (back-donation)
    t_ct2 = 0.7 * contact * (0.4 + 0.6 * polar)        # LH <-> PL  (ligand -> pocket CT)

    # --- inter-site contact stabilization ---------------------------------
    contact_stab = 0.30 * min(hbond, 4.0) / 4.0 + 0.25 * min(hydro, 20.0) / 20.0 + 0.25 * electro
    if chelation_relevant:
        contact_stab += 0.35 * metal + 0.10 * min(n_chel, 4.0) / 4.0
    contact_stab = min(contact_stab, _MAX_TOTAL_V)
    V = {
        (LH, PH): -0.35 * contact_stab,
        (LH, PL): -0.20 * contact_stab,
        (LL, PH): -0.20 * contact_stab,
        (LL, PL): -0.25 * contact_stab,
    }

    # --- assemble the 16x16 matrix ----------------------------------------
    n = N_ORBITALS
    num = [number(n, i) for i in range(n)]
    H = sum(eps[k] * num[o] for o, k in enumerate(["LH", "LL", "PH", "PL"]))
    H = H + t_pol * (_hop(n, LH, LL) + _hop(n, PH, PL))
    H = H + t_ct * _hop(n, LL, PH) + t_ct2 * _hop(n, LH, PL)
    for (i, j), v in V.items():
        H = H + v * (num[i] @ num[j])

    # --- qiskit Pauli decomposition ----------------------------------------
    from qiskit.quantum_info import SparsePauliOp

    pauli_op = SparsePauliOp.from_operator(H).simplify()

    # --- exact reference (N=2 sector) --------------------------------------
    sector_idx = occupation_indices_with(n, 2)
    sector = H[np.ix_(sector_idx, sector_idx)]
    evals, evecs = np.linalg.eigh(sector)
    exact_energy = float(evals[0])
    ground_sector = evecs[:, 0]

    full_state = np.zeros(2 ** n, dtype=complex)
    for amp, idx in zip(ground_sector, sector_idx):
        full_state[idx] = amp

    # sanity: the unrestricted ground state should also live in N=2
    full_evals = np.linalg.eigvalsh(H)
    particle_number_ok = bool(
        abs(full_evals[0] - exact_energy) < 1e-6
        or np.linalg.norm(total_number(n) @ full_state - 2.0 * full_state) < 1e-6
    )

    decoupled = eps["LH"] + eps["PH"]

    return BindingModel(
        eps=eps,
        t_pol=t_pol,
        t_ct=t_ct,
        t_ct2=t_ct2,
        V=V,
        matrix=np.asarray(H),
        pauli_op=pauli_op,
        exact_energy=exact_energy,
        exact_ground_state=full_state,
        decoupled_energy=decoupled,
        particle_number_ok=particle_number_ok,
    )


def sector_exact_energy(matrix: np.ndarray, n_electrons: int = 2) -> float:
    """Ground energy of a Hamiltonian restricted to an N-electron sector."""
    idx = occupation_indices_with(N_ORBITALS, n_electrons)
    sector = matrix[np.ix_(idx, idx)]
    return float(np.linalg.eigvalsh(sector)[0])
