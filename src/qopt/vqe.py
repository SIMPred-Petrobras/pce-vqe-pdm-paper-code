"""Variational quantum eigensolver (VQE) with a Hamming-weight-preserving ansatz.

One qubit per binary variable: the energy of a bitstring is the QUBO cost,
and the VQE prepares a parameterized state whose expected energy is
minimized. The ansatz here never leaves the subspace of bitstrings with
exactly ``k`` ones, so a cardinality constraint "select exactly ``k`` of ``n``"
holds for every measured bitstring without any penalty term.

Circuit
-------
* **Register ``x``** (``n`` qubits, one per item): it starts in
  ``|1...1 0...0>`` (``k`` ones, prepared with ``X`` gates) and goes through a
  *pyramid* of reconfigurable beam splitter (RBS) gates on neighboring qubits
  (Landman et al., Quantum 6, 881, 2022). An RBS gate of angle ``theta`` only
  mixes ``|01>`` and
  ``|10>``::

      |01> -> cos(theta) |01> - sin(theta) |10>
      |10> -> sin(theta) |01> + cos(theta) |10>

  and leaves ``|00>`` and ``|11>`` unchanged, so it preserves the number of
  ones. The pyramid has ``n (n - 1) / 2`` gates (120 for ``n = 16``).
* **Free qubit ``y``** (optional): one ``RY(phi)`` rotation; it carries an
  auxiliary variable that is not part of the cardinality constraint (here,
  "the FFT is computed").

Simulation
----------
The state is simulated exactly in the subspace of weight ``k``
(``C(n, k)`` amplitudes, 4368 for ``n = 16, k = 5``), times the two states of
``y``. All gates are real rotations, so amplitudes stay real. Because the
QUBO is diagonal in the computational basis, the expected energy is
``sum_s p(s) E(s)``, and a measurement samples bitstrings with probabilities
``p(s)``. This equals ``<psi| H |psi>`` for the Ising Hamiltonian of the
article (``H |x> = (E(x) - C) |x>``) up to the constant ``C``, so the
Hamiltonian never needs to be built. The gradient is computed exactly by the
adjoint method.

Training
--------
* ``"adam"`` -- exact expected energy and exact gradient (simulation);
* ``"spsa"`` -- expected energy estimated from ``training_shots`` samples,
  as on hardware; two estimates per iteration;
* ``"cobyla"`` -- the derivative-free COBYLA method of SciPy on the exact
  expected energy.

At the end, ``readout_shots`` bitstrings are sampled and the one with the
lowest QUBO energy is the answer. An optional ``accept`` test (for example,
an inequality constraint encoded only approximately in the QUBO) restricts
the answer to the accepted samples, if any was measured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from .qubo import QUBO


def pyramid(n: int) -> list[int]:
    """Gates of the RBS pyramid on ``n`` qubits, in time order; gate ``i`` acts on qubits ``(i, i + 1)``.

    Gate ``(i, i+1)`` appears at the time steps ``t = i, i + 2, ..., 2n - 4 - i``,
    so qubit pair ``i`` gets ``n - 1 - i`` gates and the total is ``n (n - 1) / 2``.
    """
    gates = []
    for t in range(2 * n - 3):
        gates += [i for i in range(n - 1) if i % 2 == t % 2 and i <= t <= 2 * n - 4 - i]
    return gates


@dataclass
class VQEConfig:
    """Hyper-parameters of the VQE.

    Attributes:
        optimizer: ``"adam"``, ``"spsa"`` or ``"cobyla"``.
        training_shots: Shots per loss estimate (``None`` = exact energy).
        learning_rate: Adam step size.
        max_steps: Largest number of Adam steps.
        patience, tolerance: Adam stops when the loss improved by less than
            ``tolerance`` (relative) over the last ``patience`` steps.
        spsa_iterations: SPSA iterations (two loss estimates each).
        spsa_c: Initial SPSA perturbation (radians).
        spsa_target_step: Size of the first SPSA update (radians), which calibrates its gain.
        cobyla_maxiter: Largest number of COBYLA loss evaluations.
        cobyla_rhobeg: Initial COBYLA step (radians).
        readout_shots: Bitstrings sampled at the end.
        log_every: Record the exact expected energy every ``log_every`` iterations.
    """

    optimizer: str = "adam"
    training_shots: int | None = None
    learning_rate: float = 0.05
    max_steps: int = 3000
    patience: int = 50
    tolerance: float = 1e-6
    spsa_iterations: int = 3000
    spsa_c: float = 0.2
    spsa_target_step: float = 0.3
    cobyla_maxiter: int = 3000
    cobyla_rhobeg: float = 0.5
    readout_shots: int = 1024
    log_every: int = 10


class HammingWeightCircuit:
    """Exact simulation of the Hamming-weight-preserving ansatz (see the module docstring).

    Args:
        n: Qubits of the register ``x``.
        k: Number of ones (the Hamming weight).
        free_qubit: Add the free qubit ``y`` (one ``RY`` angle, the last parameter).
    """

    def __init__(self, n: int, k: int, free_qubit: bool = True):
        self.n, self.k, self.free_qubit = n, k, free_qubit
        combos = list(combinations(range(n), k))
        self.basis = np.zeros((len(combos), n), dtype=np.int8)        # bitstrings of weight k
        for r, idx in enumerate(combos):
            self.basis[r, list(idx)] = 1
        index = {row.tobytes(): r for r, row in enumerate(self.basis)}
        self.gates = pyramid(n)
        self.n_params = len(self.gates) + int(free_qubit)
        # For each qubit pair i: the basis states with (x_i, x_{i+1}) = (0, 1) and their partners with (1, 0).
        self._pairs = []
        for i in range(n - 1):
            a = np.flatnonzero((self.basis[:, i] == 0) & (self.basis[:, i + 1] == 1))
            partner = self.basis[a].copy()
            partner[:, [i, i + 1]] = partner[:, [i + 1, i]]
            b = np.array([index[row.tobytes()] for row in partner], dtype=int)
            self._pairs.append((a, b))
        self.start = index[np.r_[np.ones(k, dtype=np.int8), np.zeros(n - k, dtype=np.int8)].tobytes()]

    @property
    def dim(self) -> int:
        """Number of bitstrings of weight ``k``."""
        return len(self.basis)

    def _forward(self, theta: np.ndarray) -> np.ndarray:
        psi = np.zeros(self.dim)
        psi[self.start] = 1.0
        for g, i in enumerate(self.gates):
            a, b = self._pairs[i]
            c, s = np.cos(theta[g]), np.sin(theta[g])
            pa, pb = psi[a], psi[b]
            psi[a], psi[b] = c * pa - s * pb, s * pa + c * pb
        return psi

    def p_one(self, params: np.ndarray) -> float:
        """Probability that the free qubit is measured as 1."""
        return float(np.sin(params[-1] / 2) ** 2) if self.free_qubit else 0.0

    def probabilities(self, params: np.ndarray) -> np.ndarray:
        """Probability of every outcome, shape ``(dim, 2)`` (column = value of ``y``; ``(dim, 1)`` without it)."""
        px = self._forward(params[:len(self.gates)]) ** 2
        if not self.free_qubit:
            return px[:, None]
        p1 = self.p_one(params)
        return np.c_[px * (1 - p1), px * p1]

    def outcomes(self) -> np.ndarray:
        """Bitstrings of all outcomes, in the order of ``probabilities(...).ravel()``: ``(x, y)`` rows."""
        if not self.free_qubit:
            return self.basis.copy()
        rows = np.repeat(self.basis, 2, axis=0)
        return np.c_[rows, np.tile([0, 1], self.dim)].astype(np.int8)

    def value_and_grad(self, params: np.ndarray, energies: np.ndarray) -> tuple[float, np.ndarray]:
        """Expected energy and its exact gradient (adjoint method).

        Args:
            params: Circuit parameters.
            energies: Energy of every outcome, shape ``(dim, 2)`` (or ``(dim, 1)``).
        """
        theta = params[:len(self.gates)]
        psi = self._forward(theta)
        if self.free_qubit:
            p1 = self.p_one(params)
            e = (1 - p1) * energies[:, 0] + p1 * energies[:, 1]
        else:
            e = energies[:, 0]
        value = float(psi ** 2 @ e)
        grad = np.zeros(self.n_params)
        if self.free_qubit:                                          # d p1 / d phi = sin(phi) / 2
            grad[-1] = float(psi ** 2 @ (energies[:, 1] - energies[:, 0])) * np.sin(params[-1]) / 2
        lam = 2.0 * e * psi
        for g in range(len(self.gates) - 1, -1, -1):
            a, b = self._pairs[self.gates[g]]
            c, s = np.cos(theta[g]), np.sin(theta[g])
            qa, qb = psi[a], psi[b]
            pa, pb = c * qa + s * qb, -s * qa + c * qb                 # state before the gate
            grad[g] = float(lam[a] @ (-s * pa - c * pb) + lam[b] @ (c * pa - s * pb))
            psi[a], psi[b] = pa, pb
            la, lb = lam[a], lam[b]
            lam[a], lam[b] = c * la + s * lb, -s * la + c * lb
        return value, grad

    def sample(self, params: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Counts of every outcome in ``shots`` measurements, shape ``(dim * 2,)`` (or ``(dim,)``)."""
        p = self.probabilities(params).ravel()
        return rng.multinomial(shots, p / p.sum())


@dataclass
class VQEResult:
    """Outcome of one VQE run.

    Attributes:
        bits: The answer ``(x, y)``: the sampled bitstring with the lowest energy (among the accepted ones,
            when an ``accept`` test is given and some sample passes it).
        accepted: Whether the answer passes the ``accept`` test (``True`` without a test).
        energy: Its QUBO energy.
        loss: Exact expected energy of the final state.
        p_answer: Exact probability of measuring the answer's ``x`` (any ``y``).
        most_likely: Most probable bitstring ``(x, y)`` of the final state.
        steps, evaluations, shots_used: Iterations, energy evaluations and training shots.
        params: Final parameters.
        history: Exact expected energy every ``log_every`` iterations.
        seconds: Run time.
    """

    bits: np.ndarray
    accepted: bool
    energy: float
    loss: float
    p_answer: float
    most_likely: np.ndarray
    steps: int
    evaluations: int
    shots_used: int
    params: np.ndarray
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def run_vqe(qubo: QUBO, k: int, config: VQEConfig, seed=0, progress=None, accept=None) -> VQEResult:
    """Minimize a QUBO whose first ``qubo.n - 1`` variables must have exactly ``k`` ones (the last is free).

    The free last variable is the ``y`` qubit. The initial parameters are
    uniform in ``[-pi, pi)`` and depend only on ``seed``, so every optimizer
    starts from the same point for the same seed. ``accept`` (optional) maps
    bitstrings ``(m, n + 1)`` to booleans; the readout then keeps the
    lowest-energy accepted sample.
    """
    t0 = time.time()
    n = qubo.n - 1
    circuit = HammingWeightCircuit(n, k, free_qubit=True)
    outcomes = circuit.outcomes()
    energies = np.asarray(qubo.energy(outcomes), dtype=float).reshape(circuit.dim, 2)
    flat = energies.ravel()
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    params = rng.uniform(-np.pi, np.pi, circuit.n_params)
    history, counter = [], {"evals": 0, "shots": 0}
    expected = config.max_steps if config.optimizer == "adam" else (
        config.spsa_iterations if config.optimizer == "spsa" else config.cobyla_maxiter)

    def exact(p):
        return float(circuit.probabilities(p).ravel() @ flat)

    def estimate(p):
        counter["evals"] += 1
        if config.training_shots is None:
            return exact(p)
        counter["shots"] += config.training_shots
        return float(circuit.sample(p, config.training_shots, rng) @ flat) / config.training_shots

    def log(step, p):
        if config.log_every and step % config.log_every == 0:
            history.append({"iteration": step, "loss": exact(p)})
        if progress is not None and step % 50 == 0:
            progress(min(step / max(expected, 1), 0.99))

    steps = 0
    if config.optimizer == "adam":
        m, v = np.zeros_like(params), np.zeros_like(params)
        best, best_params, losses = np.inf, params.copy(), []
        for step in range(1, config.max_steps + 1):
            value, grad = circuit.value_and_grad(params, energies)
            counter["evals"] += 1
            losses.append(value)
            if value < best:
                best, best_params = value, params.copy()
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            params = params - config.learning_rate * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
            log(step, params)
            steps = step
            if step > config.patience:
                old = losses[-config.patience - 1]
                if old - value < config.tolerance * max(abs(old), 1e-12):
                    break
        params = best_params
    elif config.optimizer == "spsa":
        K, A = config.spsa_iterations, 0.1 * config.spsa_iterations
        c0 = config.spsa_c
        # Gain a: the first step has the target size (mean of a few gradient estimates).
        g0 = []
        for _ in range(5):
            delta = rng.choice([-1.0, 1.0], circuit.n_params)
            g0.append(abs(estimate(params + c0 * delta) - estimate(params - c0 * delta)) / (2 * c0))
        a = config.spsa_target_step * (A + 1) ** 0.602 / max(float(np.mean(g0)), 1e-12)
        for step in range(1, K + 1):
            ak, ck = a / (step + A) ** 0.602, c0 / step ** 0.101
            delta = rng.choice([-1.0, 1.0], circuit.n_params)
            diff = estimate(params + ck * delta) - estimate(params - ck * delta)
            params = params - ak * diff / (2 * ck) * delta
            log(step, params)
            steps = step
    elif config.optimizer == "cobyla":
        it = {"n": 0}

        def fun(p):
            it["n"] += 1
            log(it["n"], p)
            return estimate(p)

        res = minimize(fun, params, method="COBYLA",
                       options={"maxiter": config.cobyla_maxiter, "rhobeg": config.cobyla_rhobeg})
        params, steps = np.asarray(res.x, dtype=float), it["n"]
    else:
        raise ValueError(f"unknown optimizer {config.optimizer!r}; use 'adam', 'spsa' or 'cobyla'")

    # Readout: sample bitstrings and keep the one with the lowest QUBO energy (among the accepted ones).
    counts = circuit.sample(params, config.readout_shots, rng)
    seen = np.flatnonzero(counts)
    ok = np.ones(len(seen), dtype=bool) if accept is None else np.asarray(accept(outcomes[seen]), dtype=bool)
    pool = seen[ok] if ok.any() else seen
    best = pool[np.argmin(flat[pool])]
    probs = circuit.probabilities(params)
    answer = outcomes[best]
    if progress is not None:
        progress(1.0)
    return VQEResult(bits=answer, accepted=bool(ok.any()), energy=float(flat[best]), loss=exact(params),
                     p_answer=float(probs[best // 2].sum()), most_likely=outcomes[int(np.argmax(probs.ravel()))],
                     steps=steps, evaluations=counter["evals"], shots_used=counter["shots"],
                     params=params, history=history, seconds=time.time() - t0)
