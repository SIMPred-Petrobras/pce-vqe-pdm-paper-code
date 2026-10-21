"""Estimated run time of the PCE runs, used to weight the progress bar and by ``--plan``.

The time of one optimizer iteration of the 16-qubit PCE circuit depends on
the machine, so it is measured at start-up on the actual circuit
(:func:`calibrate`, a few seconds). Estimates are only used to display
progress: nothing in the results depends on them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from qopt.pce import StateVectorCircuit, min_qubits, pauli_strings

#: Seconds to read one measurement from its archive and compute the features of its windows.
MEASUREMENT_SECONDS = 0.5
#: Circuits with at least this many qubits are limited by memory bandwidth when
#: all workers run at once (their state vectors no longer fit in the caches).
LARGE_CIRCUIT = 12


@dataclass
class Speeds:
    """Measured seconds per exact gradient (``grad``) and per loss evaluation (``eval``), per circuit size.

    Attributes:
        grad, eval: Single-process timings, keyed by ``(qubits, layers)``.
        load: Slowdown of large circuits when every worker runs one at the
            same time (memory bandwidth is shared); measured with the worker
            pool, ``1`` until then.
    """

    grad: dict[tuple[int, int], float] = field(default_factory=dict)
    eval: dict[tuple[int, int], float] = field(default_factory=dict)
    load: float = 1.0

    def get(self, kind: str, n: int, layers: int) -> float:
        """Seconds per iteration under full load."""
        factor = self.load if n >= LARGE_CIRCUIT else 1.0
        table = self.grad if kind == "grad" else self.eval
        if (n, layers) in table:
            return table[(n, layers)] * factor
        # Unmeasured size: scale the closest measurement by 2^n and by the number of layers.
        if not table:
            return 1e-3
        (n0, l0), t0 = min(table.items(), key=lambda kv: abs(kv[0][0] - n) + abs(kv[0][1] - layers))
        return t0 * 2.0 ** (n - n0) * layers / l0 * factor


def calibrate(sizes: set[tuple[int, int, int]], repeats: int = 2, ansatz: str = "heavy_hex") -> Speeds:
    """Time one exact gradient and one shot-based loss evaluation for every ``(m, n_qubits, layers)``."""
    speeds = Speeds()
    rng = np.random.default_rng(0)
    for m, n, layers in sorted(sizes):
        circuit = StateVectorCircuit(n, layers, pauli_strings(m, 2, n), ansatz)
        params = rng.uniform(-np.pi, np.pi, circuit.n_params)
        w = rng.normal(size=m)
        loss = lambda c: (float(w @ c), w)
        circuit.value_and_grad(params, loss)                     # warm-up
        t0 = time.perf_counter()
        for _ in range(repeats):
            circuit.value_and_grad(params, loss)
        speeds.grad[(n, layers)] = (time.perf_counter() - t0) / repeats
        t0 = time.perf_counter()
        for _ in range(repeats):
            circuit.estimate(params, 1024, rng)
        speeds.eval[(n, layers)] = (time.perf_counter() - t0) / repeats
    return speeds


def qubits(m: int, pce) -> int:
    """Qubits of the PCE circuit for ``m`` variables."""
    return pce.n_qubits or min_qubits(m, pce.k)
