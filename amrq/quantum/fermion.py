"""Minimal fermionic algebra over the Jordan-Wigner (JW) occupation basis.

Qubit ``i`` encodes the occupation of spin-orbital ``i``. A computational
basis state index encodes occupations as |q_{n-1} ... q_1 q_0> (Qiskit's
little-endian convention), i.e. bit ``i`` of the index is orbital i's
occupation.  The JW string sign of annihilating orbital ``i`` is
(-1)^(number of occupied orbitals with index < i), which is exactly what
``_jw_parity`` computes.

This is intentionally dependency-light: the operators are dense 2^n x 2^n
matrices, which is fine for the 4-orbital binding model used by AMR-Q.
"""
from __future__ import annotations

import numpy as np


def jw_parity(index: int, orbital: int) -> int:
    """Parity of occupied orbitals with index < ``orbital`` (JW sign)."""
    mask = (1 << orbital) - 1
    return bin(index & mask).count("1") % 2


def annihilation(n_orbitals: int, orbital: int) -> np.ndarray:
    """Matrix of a_i in the JW occupation basis."""
    dim = 2 ** n_orbitals
    mat = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        if (col >> orbital) & 1:
            sign = 1.0 if jw_parity(col, orbital) == 0 else -1.0
            mat[col ^ (1 << orbital), col] = sign
    return mat


def creation(n_orbitals: int, orbital: int) -> np.ndarray:
    """Matrix of a_i^dagger (Hermitian conjugate of annihilation)."""
    return annihilation(n_orbitals, orbital).conj().T


def number(n_orbitals: int, orbital: int) -> np.ndarray:
    """Diagonal number operator n_i = a_i^dagger a_i."""
    dim = 2 ** n_orbitals
    return np.diag([float((idx >> orbital) & 1) for idx in range(dim)]).astype(complex)


def total_number(n_orbitals: int) -> np.ndarray:
    """N = sum_i n_i."""
    return sum(number(n_orbitals, i) for i in range(n_orbitals))


def occupation_indices_with(n_orbitals: int, n_electrons: int) -> list[int]:
    """All basis indices with exactly ``n_electrons`` occupied orbitals."""
    from itertools import combinations

    out = []
    for occs in combinations(range(n_orbitals), n_electrons):
        idx = 0
        for o in occs:
            idx |= 1 << o
        out.append(idx)
    return sorted(out)
