"""Variational Quantum Eigensolver for the AMR-Q binding Hamiltonian.

Ansatz: Hartree-Fock reference |LH,PH> (X on qubits 0 and 2) followed by R
layers of single-excitation (Givens) gates on the pairs (0,1), (2,3), (0,3),
(1,2). Each SingleExcitation(theta) is implemented as

    CX(i -> j) . CRY(theta; control j, target i) . CX(i -> j)

which realizes exp(-i theta/2 (X_i Y_j - Y_i X_j)) -- a *particle-conserving*
rotation in the {|10>, |01>} subspace of the two qubits. Number conservation
is exact by construction, so the same circuit runs unchanged on real IBM
Quantum hardware.

Optimizers: scipy COBYLA (default, gradient-free) or a built-in SPSA
(simultaneous perturbation stochastic approximation), the classic
quantum-native optimizer designed for noisy hardware.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

EXCITATION_PAIRS = [(0, 1), (2, 3), (0, 3), (1, 2)]
HF_OCCUPIED = [0, 2]


def build_ansatz(reps: int = 2):
    """Parametric VQE ansatz circuit + ordered parameter list."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    n_params = reps * len(EXCITATION_PAIRS)
    params = ParameterVector("theta", n_params)
    qc = QuantumCircuit(4, name="AMR-Q-ansatz")
    for q in HF_OCCUPIED:
        qc.x(q)
    k = 0
    for _ in range(reps):
        for (i, j) in EXCITATION_PAIRS:
            qc.cx(i, j)
            qc.cry(params[k], j, i)
            qc.cx(i, j)
            k += 1
    return qc, list(params)


def _excitation_matrix(theta: float, i: int, j: int) -> np.ndarray:
    """Unitary of SingleExcitation(theta) on 2 qubits placed on wires i<j."""
    dim = 16
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    u = np.eye(dim, dtype=complex)
    bi, bj = 1 << i, 1 << j
    for idx in range(dim):
        if (idx & bi) and not (idx & bj):        # |i occ, j vac> -> mix
            partner = idx ^ bi ^ bj
            u[idx, idx] = c
            u[partner, idx] = -s
            u[idx, partner] = s
            u[partner, partner] = c
    return u


def exact_reference_energy(matrix: np.ndarray, n_electrons: int = 2) -> float:
    from .model import sector_exact_energy

    return sector_exact_energy(matrix, n_electrons)


@dataclass
class VQEResult:
    energy: float
    params: np.ndarray
    trace: list[float] = field(default_factory=list)
    iterations: int = 0
    optimizer: str = "cobyla"
    converged: bool = False
    exact_energy: float = 0.0
    ground_state: np.ndarray = field(default_factory=lambda: np.empty(0))
    ct_weight: float = 0.0
    number_expectation: float = 2.0

    @property
    def error(self) -> float:
        return abs(self.energy - self.exact_energy)

    def to_meta(self) -> dict:
        return {
            "energy": round(self.energy, 6),
            "exact_energy": round(self.exact_energy, 6),
            "error": round(self.error, 6),
            "iterations": self.iterations,
            "optimizer": self.optimizer,
            "converged": self.converged,
            "ct_weight": round(self.ct_weight, 4),
            "number_expectation": round(self.number_expectation, 4),
            "trace": [round(float(e), 6) for e in self.trace],
        }


def run_vqe(
    model,
    reps: int = 2,
    optimizer: str = "cobyla",
    maxiter: int = 400,
    seed: int = 0,
    evaluator=None,
    keep_trace_limit: int = 4000,
) -> VQEResult:
    """Optimize the ansatz to minimize <H>. ``evaluator(qc, x, op) -> float``
    abstracts the quantum backend (see backends.py); defaults to exact
    statevector evaluation."""
    qc, params = build_ansatz(reps=reps)
    op = model.pauli_op
    evaluate = evaluator or _default_evaluator

    rng = np.random.default_rng(seed)
    x0 = 0.1 * rng.standard_normal(len(params))

    trace: list[float] = []
    best = [np.inf, x0.copy()]

    def objective(x):
        energy = float(evaluate(qc, np.asarray(x, dtype=float), op))
        trace.append(energy)
        if energy < best[0]:
            best[0], best[1] = energy, np.asarray(x, dtype=float).copy()
        return energy

    if optimizer == "spsa":
        x_opt = _spsa(objective, x0, maxiter=maxiter, rng=rng)
        iterations = maxiter
    else:
        from scipy.optimize import minimize

        res = minimize(
            objective,
            x0,
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": 0.6, "tol": 1e-5},
        )
        x_opt = np.asarray(res.x, dtype=float)
        iterations = int(res.get("nfev", len(trace)))

    energy = float(evaluate(qc, x_opt, op))
    if energy > best[0] + 1e-9:  # keep the best point seen
        energy, x_opt = best[0], best[1]

    # final statevector diagnostics (exact, independent of backend)
    bound = qc.assign_parameters(x_opt)
    from qiskit.quantum_info import Statevector

    sv = Statevector(bound).data
    probs = np.abs(sv) ** 2
    number_exp = float(sum(p * bin(i).count("1") for i, p in enumerate(probs)))

    # convergence: tiny spread over the tail of the trace
    tail = trace[-25:] if len(trace) >= 25 else trace
    converged = bool(tail and (max(tail) - min(tail)) < 1e-3)

    result = VQEResult(
        energy=energy,
        params=x_opt,
        trace=trace[:keep_trace_limit],
        iterations=iterations,
        optimizer=optimizer,
        converged=converged,
        exact_energy=model.exact_energy,
        ground_state=sv,
        ct_weight=model.ct_weight(sv),
        number_expectation=number_exp,
    )
    return result


def _default_evaluator(qc, x, op) -> float:
    """Exact statevector expectation value (fast, deterministic)."""
    from qiskit.quantum_info import Statevector

    sv = Statevector(qc.assign_parameters(x))
    return float(np.real(sv.expectation_value(op)))


def _spsa(objective, x0, maxiter, rng, a: float = 0.35, c: float = 0.18,
          alpha: float = 0.602, gamma: float = 0.101) -> np.ndarray:
    """Simultaneous Perturbation Stochastic Approximation (Spall, 1992)."""
    x = np.asarray(x0, dtype=float).copy()
    A = 0.1 * maxiter
    for k in range(maxiter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=x.shape)
        fp = objective(x + ck * delta)
        fm = objective(x - ck * delta)
        ghat = (fp - fm) / (2.0 * ck) * delta
        x = x - ak * ghat
    return x


def stable_seed(*parts) -> int:
    """Deterministic seed from arbitrary strings (reproducible runs)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)
