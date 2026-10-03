"""Quantum evaluation backends for the VQE loop.

Every backend implements ``evaluate(qc, params, observable) -> float`` so the
VQE loop is backend-agnostic:

- ``statevector`` : exact dense simulation with qiskit.quantum_info.Statevector.
                    This is the free IBM Quantum *simulator* path -- fast,
                    deterministic, always available. Default.
- ``estimator``   : qiskit.primitives.StatevectorEstimator (the Qiskit V2
                    primitives API -- the same interface used for real hardware).
- ``aer``         : qiskit-aer shot-based simulator, optionally with a noise
                    model (requires the optional ``qiskit-aer`` package).
- ``ibm``         : real IBM Quantum backend via qiskit-ibm-runtime (requires
                    the optional package + a free API token in AMRQ_IBM_TOKEN
                    or IBM_QUANTUM_TOKEN).
"""
from __future__ import annotations

import os

import numpy as np


class BackendInfo:
    def __init__(self, key: str, label: str, available: bool, detail: str = ""):
        self.key, self.label, self.available, self.detail = key, label, available, detail


def available_backends() -> list[BackendInfo]:
    infos = [
        BackendInfo("statevector", "Exact statevector simulator (free, default)", True),
        BackendInfo("estimator", "Qiskit StatevectorEstimator (V2 primitives)", True),
    ]
    try:
        import qiskit_aer  # noqa: F401

        infos.append(BackendInfo("aer", "Aer shot-based simulator (noisy-capable)", True))
    except Exception:
        infos.append(BackendInfo("aer", "Aer simulator (install qiskit-aer)", False,
                                 "pip install qiskit-aer"))
    token = os.environ.get("AMRQ_IBM_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
    try:
        import qiskit_ibm_runtime  # noqa: F401

        infos.append(BackendInfo(
            "ibm", "IBM Quantum hardware (runtime)", bool(token),
            "" if token else "Set AMRQ_IBM_TOKEN to enable (free account at quantum.ibm.com)"))
    except Exception:
        infos.append(BackendInfo("ibm", "IBM Quantum hardware", False,
                                 "pip install qiskit-ibm-runtime and set AMRQ_IBM_TOKEN"))
    return infos


def make_evaluator(backend: str = "statevector", shots: int = 4096):
    """Return an ``evaluate(qc, params, observable) -> float`` callable."""
    backend = (backend or "statevector").lower()

    if backend == "statevector":
        from .vqe import _default_evaluator

        return _default_evaluator

    if backend == "estimator":
        from qiskit.primitives import StatevectorEstimator

        estimator = StatevectorEstimator()

        def evaluate(qc, params, observable):
            job = estimator.run([(qc, observable, np.asarray(params, dtype=float))])
            return float(np.asarray(job.result()[0].data.evs).reshape(-1)[0])

        return evaluate

    if backend == "aer":
        try:
            from qiskit_aer.primitives import EstimatorV2
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "Aer backend requested but qiskit-aer is not installed. "
                "pip install qiskit-aer") from exc
        estimator = EstimatorV2(options={"default_precision": 1.0 / shots})

        def evaluate(qc, params, observable):
            job = estimator.run([(qc, observable, np.asarray(params, dtype=float))])
            return float(np.asarray(job.result()[0].data.evs).reshape(-1)[0])

        return evaluate

    if backend == "ibm":
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, Batch
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "IBM backend requested but qiskit-ibm-runtime is not installed. "
                "pip install qiskit-ibm-runtime") from exc
        token = os.environ.get("AMRQ_IBM_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            raise RuntimeError(
                "IBM backend requested but no token found. Set the AMRQ_IBM_TOKEN "
                "environment variable (free account at https://quantum.ibm.com).")
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        backend_obj = service.least_busy(operational=True, simulator=False)
        estimator = EstimatorV2(mode=backend_obj)

        def evaluate(qc, params, observable):
            with Batch(mode=backend_obj) as batch:
                est = EstimatorV2(mode=batch)
                job = est.run([(qc, observable, np.asarray(params, dtype=float))])
                return float(np.asarray(job.result()[0].data.evs).reshape(-1)[0])

        return evaluate

    raise ValueError(f"Unknown backend: {backend!r}. "
                     f"Choose from: statevector, estimator, aer, ibm.")
