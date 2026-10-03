"""Tests for the quantum layer: fermion algebra, binding model, VQE."""
import numpy as np
import pytest

from amrq.quantum.fermion import (annihilation, creation, number,
                                  occupation_indices_with, total_number)
from amrq.quantum.model import HF_STATE_INDEX, N_ORBITALS, build_binding_model
from amrq.quantum.vqe import build_ansatz, run_vqe, stable_seed


@pytest.fixture
def features():
    return dict(max_negative_charge=0.45, max_positive_charge=0.30,
                contact_frac=0.6, hbond_contacts=2, hydrophobic_contacts=10,
                electrostatic_complementarity=0.3, metal_proximity_score=0.0,
                chelating_atoms=0, pocket_polar_frac=0.45, chelation_relevant=False)


@pytest.fixture
def model(features):
    return build_binding_model(features)


def test_fermion_operators_anticommute():
    n = 4
    a0, a1 = annihilation(n, 0), annihilation(n, 1)
    ad0 = creation(n, 0)
    # {a_i, a_j^dagger} = delta_ij
    assert np.allclose(a0 @ ad0 + ad0 @ a0, np.eye(16))
    assert np.allclose(a0 @ a1 + a1 @ a0, np.zeros((16, 16)))


def test_number_operator_matches_occupation_count():
    n = 4
    N = total_number(n)
    assert N[HF_STATE_INDEX, HF_STATE_INDEX] == 2  # HF has LH and PH occupied


def test_hamiltonian_is_hermitian_and_particle_conserving(model):
    assert np.allclose(model.matrix, model.matrix.conj().T)
    N = total_number(N_ORBITALS)
    comm = model.matrix @ N - N @ model.matrix
    assert np.abs(comm).max() < 1e-10


def test_pauli_decomposition_reconstructs_matrix(model):
    rebuilt = model.pauli_op.to_matrix()
    assert np.allclose(rebuilt, model.matrix, atol=1e-10)


def test_decoupled_energy_is_hf_reference(model):
    assert model.decoupled_energy == pytest.approx(model.eps["LH"] + model.eps["PH"])


def test_coupling_stabilizes_ground_state(model):
    assert model.delta_e_ev <= 1e-9  # hopping/contact terms can only lower E0


def test_exact_ground_state_in_n2_sector(model):
    probs = np.abs(model.exact_ground_state) ** 2
    for idx, p in enumerate(probs):
        if p > 1e-12:
            assert bin(idx).count("1") == 2


def test_ansatz_parameter_count():
    qc, params = build_ansatz(reps=2)
    assert len(params) == 8
    assert qc.num_qubits == 4


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_vqe_converges_to_exact_energy(model, seed):
    result = run_vqe(model, reps=2, optimizer="cobyla", maxiter=400, seed=seed)
    assert result.error < 1e-3
    assert result.number_expectation == pytest.approx(2.0, abs=1e-6)
    assert 0.0 <= result.ct_weight <= 1.0
    assert result.converged


def test_vqe_spsa_reaches_exact_energy(model):
    result = run_vqe(model, reps=2, optimizer="spsa", maxiter=250, seed=3)
    assert result.error < 5e-3
    assert result.number_expectation == pytest.approx(2.0, abs=1e-6)


def test_vqe_energy_matches_estimator_backend(model):
    from amrq.quantum.backends import make_evaluator

    result = run_vqe(model, reps=2, optimizer="cobyla", maxiter=200, seed=5)
    evaluator = make_evaluator("estimator")
    from amrq.quantum.vqe import build_ansatz as ba

    qc, _ = ba(reps=2)
    energy = evaluator(qc, result.params, model.pauli_op)
    assert energy == pytest.approx(result.energy, abs=1e-6)


def test_metal_chelation_strengthens_binding():
    base = dict(max_negative_charge=0.4, max_positive_charge=0.2,
                contact_frac=0.5, hbond_contacts=1, hydrophobic_contacts=6,
                electrostatic_complementarity=0.2, chelating_atoms=2,
                pocket_polar_frac=0.5)
    without = build_binding_model({**base, "metal_proximity_score": 0.0,
                                   "chelation_relevant": True})
    with_metal = build_binding_model({**base, "metal_proximity_score": 0.9,
                                      "chelation_relevant": True})
    # stronger chelation -> more stable coupled complex (more negative delta E)
    assert with_metal.delta_e_ev < without.delta_e_ev


def test_stable_seed_is_deterministic():
    assert stable_seed("a", 1) == stable_seed("a", 1)
    assert stable_seed("a", 1) != stable_seed("a", 2)


def test_sector_indices_count():
    assert len(occupation_indices_with(4, 2)) == 6
