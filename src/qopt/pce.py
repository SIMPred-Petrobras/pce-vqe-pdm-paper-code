"""Pauli-correlation encoding (PCE): a qubit-efficient variational solver for QUBO problems.

This module implements the quantum optimizer of Sciorilli et al., "Towards
large-scale quantum optimization solvers with few qubits", Nature
Communications 16, 476 (2025), extended to QUBO problems whose variables are
split into blocks with cardinality constraints and free blocks
(:class:`qopt.feasible.Block`).

Encoding
--------
Each binary variable is encoded in the sign of the expectation value of one
``k``-body Pauli string over ``n`` qubits: ``x_i = (1 + sgn <Pi_i>) / 2``. The
strings are ``X^k``, ``Y^k`` and ``Z^k`` on every ``k``-subset of the qubits
(:func:`pauli_strings`), so ``3 C(n, k)`` variables fit in ``n`` qubits; with
``k = 2`` (quadratic compression), 30 variables need 5 qubits and 320 need 16.
Strings of the same axis commute, so all correlators are read from at most
three measurement settings (every qubit measured in the X, Y or Z basis).

Ansatz
------
Two architectures, both starting from ``|0...0>`` (``PCEConfig.ansatz``):

* ``"brickwork"`` (:func:`brickwork`, the PCE paper's circuit): every layer
  applies a rotation around one axis (X, Y and Z in turn, layer after layer)
  to every qubit, followed by Molmer-Sorensen gates
  ``MS(phi0, phi1, theta) = exp(-i theta/2 sigma_phi0 (x) sigma_phi1)``,
  ``sigma_phi = cos(phi) X + sin(phi) Y``, on neighboring pairs of a line,
  alternating between even and odd pairs. MS gates are native to trapped-ion
  processors.
* ``"heavy_hex"`` (:func:`heavy_hex`, IBM-native; the circuit of the
  experiment, while the brickwork was used in its first version): every layer applies ``RY``
  and ``RZ`` to every qubit, followed by ``RZZ(theta) = exp(-i theta Z (x) Z / 2)``
  gates on one set of disjoint couplings of the heavy-hex graph of IBM
  processors, cycling through the sets. ``RZZ`` is locally equivalent to the
  MS gate, and the heavy-hex graph has a smaller diameter than a line, so the
  same number of two-qubit gates correlates every pair of qubits sooner.

Loss
----
The QUBO objective, in spin form ``h . s + sum_{i<j} J_ij s_i s_j``
(``x = (1 + s) / 2``), is relaxed by replacing every binary variable with a
smooth one in ``(0, 1)``. The regularization of the paper,
``beta [(1/m) sum_i t_i^2]^2`` with ``t_i = tanh(alpha <Pi_i>)``, keeps the
correlators small, where every sign pattern can be reached. Free variables
are relaxed as ``x~_i = (1 + t_i) / 2``. Constrained blocks (exactly ``w``
ones) are relaxed in one of two ways (``PCEConfig.cardinality``):

* ``"offset"`` (default) -- a soft version of the top-``w`` decoding:
  ``x~_i = (1 + tanh(alpha (<Pi_i> - b))) / 2``, with the block offset ``b``
  chosen so that ``sum_{i in block} x~_i = w`` exactly (a one-dimensional root,
  found by bisection; the gradient goes through ``b`` by implicit
  differentiation). The relaxation satisfies the constraint by construction,
  needs no penalty and, like the decoding, depends only on the differences
  between the correlators of a block.
* ``"penalty"`` -- ``x~_i = (1 + t_i) / 2`` and the quadratic penalty
  ``rho (sum_{i in block} x~_i - w)^2``. Two-body correlators of one axis
  cannot all be strongly negative (``sum_{a<b} <P_a P_b> >= -n/2``), so when
  ``w`` is a small fraction of a large block the penalty cannot be satisfied:
  it then dominates the loss and drowns the objective. This was the first
  formulation of this work and is kept for comparison.

With ``x~ = (1 + t~) / 2``::

    L = [h . t~ + sum_{i<j} J_ij t~_i t~_j (+ rho/4 sum_blocks (sum_{i in block} t~_i - (2w - size))^2)] / nu
        + beta [(1/m) sum_i t_i^2]^2

``nu`` is half of the total absolute weight of the objective (the analog of
the paper's maximum-cut estimate for MaxCut); dividing by it changes neither
the minimizers nor the Adam steps and makes tolerances relative. This is the
loss of the article, ``sum_i Q_ii x~_i + sum_{i<j} Q_ij x~_i x~_j + beta nu
[(1/m) sum_i t_i^2]^2``, divided by ``nu`` and written in spin form (the two
differ by a constant).

Training
--------
The parameters are trained by one of three optimizers, either on an exact
state-vector simulation or from shot-based estimates (``training_shots``),
as a quantum computer would provide them:

========  ======================================  ==========================================
optimizer exact training (``training_shots=None``) shot-based training
========  ======================================  ==========================================
adam      exact gradient (adjoint method),        parameter-shift gradient estimated with
          stops on a patience criterion           shots, fixed number of steps
spsa      exact loss, two evaluations per         loss estimated with shots, two
          iteration                               evaluations per iteration
slsqp     exact loss and gradient                 loss and parameter-shift gradient
                                                  estimated with shots
========  ======================================  ==========================================

Every run records the *exact* loss of the current parameters every
``log_every`` iterations (``PCEResult.history``), so loss curves of exact and
shot-based runs are comparable.

Readout and post-processing
---------------------------
The trained circuit is measured with ``readout_shots`` shots per setting and
the correlators are estimated from the samples. Decoding generalizes
``x_i = (1 + sgn <Pi_i>) / 2`` to the constraints: in each constrained block
the ``w`` variables with the largest correlators are set to 1 (plain sign
decoding whenever the signs already satisfy the constraint); variables of
free blocks are decoded by their sign. One round of local search
(:func:`qopt.feasible.swap_search`, or a problem-specific search passed to
:func:`run_pce`) gives the answer.

Baselines and references
------------------------
Every run also records, with random streams that are identical for every
optimizer of the same problem (paired comparison): a random feasible
bitstring followed by the same round of local search; the decoded untrained
circuit, alone and followed by the search; and a steepest-descent local
search. Answers are scored against a :class:`qopt.feasible.Reference`: the
exact optimum for small problems, or the best of several classical local
searches otherwise.

Simulation
----------
:class:`StateVectorCircuit` simulates the circuit with numpy (``2**n``
amplitudes) and agrees with PennyLane to machine precision
(:func:`verify_against_pennylane`). Correlators of two-body strings are read
from the ``n x n`` matrix ``S^T diag(p) S`` of single-qubit signs ``S``, which
needs ``O(2^n n)`` memory instead of the ``O(2^n m)`` of a sign matrix per
string.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .feasible import (Block, Reference, block_slices, exact_reference, greedy_search, heuristic_reference,
                       is_feasible, random_feasible, swap_search)
from .qubo import QUBO

AXES = "XYZ"
HALF_PI = np.pi / 2

_I2 = np.eye(2, dtype=complex)
_PAULI = {"X": np.array([[0, 1], [1, 0]], dtype=complex),
          "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
_HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
#: Basis change applied to every qubit before a computational-basis measurement.
_BASIS = {"X": _HADAMARD, "Y": _HADAMARD @ np.diag([1, -1j]), "Z": _I2}

ProgressFn = Callable[[float], None]


# --------------------------------------------------------------------------
# Encoding and decoding
# --------------------------------------------------------------------------

def min_qubits(m: int, k: int) -> int:
    """Fewest qubits whose ``3 C(n, k)`` ``k``-body Pauli strings encode ``m`` variables."""
    n = max(k, 2 if k > 1 else 1)
    while 3 * comb(n, k) < m:
        n += 1
    return n


def pauli_strings(m: int, k: int, n: int) -> list[tuple[str, tuple[int, ...]]]:
    """The Pauli strings of the encoding, one per variable, as ``(axis, qubits)``.

    The strings are taken in a fixed order: every ``k``-subset of the qubits
    (lexicographic order) with ``X``, then with ``Y``, then with ``Z``.
    Variable ``i`` is encoded in string ``i``.
    """
    subsets = list(combinations(range(n), k))
    strings = [(axis, s) for axis in AXES for s in subsets]
    if m > len(strings):
        raise ValueError(f"{m} variables do not fit in 3*C({n},{k}) = {len(strings)} strings")
    return strings[:m]


def decode(correlators: np.ndarray, blocks: list[Block]) -> np.ndarray:
    """Feasible bitstring from the correlators.

    In a constrained block the ``w`` largest correlators are set to 1: this is
    sign decoding whenever the signs already satisfy the constraint, and
    otherwise flips the variables whose correlators are closest to zero. Free
    blocks are decoded by sign. Ties are broken by position.
    """
    c = np.asarray(correlators, dtype=float)
    bits = np.zeros(len(c), dtype=np.int8)
    for s, b in zip(block_slices(blocks), blocks):
        if b.constrained:
            top = np.argsort(-c[s], kind="stable")[:b.weight]
            bits[s.start + top] = 1
        else:
            bits[s] = (c[s] > 0).astype(np.int8)
    return bits


# --------------------------------------------------------------------------
# Ansatz and gate matrices
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """One parameterized gate of the ansatz.

    Attributes:
        kind: ``"R"`` (single-qubit rotation ``exp(-i theta P / 2)``), ``"MS"``
            (Molmer-Sorensen gate on two neighboring qubits) or ``"ZZ"``
            (``exp(-i theta Z (x) Z / 2)`` on any two qubits).
        axis: Rotation axis (``"X"``, ``"Y"`` or ``"Z"``); empty for two-qubit gates.
        wires: Qubits the gate acts on (for MS gates, consecutive; first one most significant).
        params: Positions of its parameters in the parameter vector
            (``(phi0, phi1, theta)`` for MS gates).
    """

    kind: str
    axis: str
    wires: tuple[int, ...]
    params: tuple[int, ...]


def brickwork(n: int, layers: int) -> tuple[list[Gate], int]:
    """Brickwork ansatz: per layer, rotations around X, Y or Z (in turn) and MS gates on alternating pairs.

    Returns:
        ``(gates, n_params)``.
    """
    gates, p = [], 0
    for layer in range(layers):
        axis = AXES[layer % 3]
        for q in range(n):
            gates.append(Gate("R", axis, (q,), (p,)))
            p += 1
        for a in range(layer % 2, n - 1, 2):
            gates.append(Gate("MS", "", (a, a + 1), (p, p + 1, p + 2)))
            p += 3
    return gates, p


def heavy_hex_edges(n: int, width: int = 7) -> list[tuple[int, int]]:
    """Couplings among the first ``n`` qubits of a heavy-hex lattice, the coupling graph of IBM processors.

    The lattice has rows of ``width`` qubits coupled in a line; consecutive
    rows are joined by "bridge" qubits at columns 0 and 4, then 2 and 6, and
    so on alternately. Qubits are numbered row 0, bridges, row 1, bridges, ...,
    so the first ``n`` qubits always form a connected subgraph. With ``n <= 7``
    the graph is a line; with ``n = 16`` it is one heavy hexagon (a 12-qubit
    ring) with two 2-qubit tails, whose diameter is 8 (a line of 16 qubits has 15).
    """
    labels, couplings, prev, row_index = [], [], None, 0
    while len(labels) < n:
        row = [("row", row_index, c) for c in range(width)]
        if prev is not None:
            columns = (0, 4) if (row_index - 1) % 2 == 0 else (2, 6)
            for c in columns:
                bridge = ("bridge", row_index, c)
                labels.append(bridge)
                couplings += [(prev[c], bridge), (bridge, row[c])]
        labels += row
        couplings += [(row[i], row[i + 1]) for i in range(width - 1)]
        prev, row_index = row, row_index + 1
    index = {q: i for i, q in enumerate(labels[:n])}
    return sorted(tuple(sorted((index[a], index[b]))) for a, b in couplings if a in index and b in index)


def edge_matchings(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Split the couplings into sets of disjoint pairs (greedy edge coloring); each set is one gate layer."""
    matchings: list[list[tuple[int, int]]] = []
    for e in edges:
        for m in matchings:
            if not any(set(e) & set(f) for f in m):
                m.append(e)
                break
        else:
            matchings.append([e])
    return matchings


def heavy_hex(n: int, layers: int) -> tuple[list[Gate], int]:
    """IBM-native ansatz: per layer, ``RY`` and ``RZ`` on every qubit and ``RZZ`` gates on one set of couplings.

    It is the brickwork architecture of the PCE paper carried over to the
    coupling graph of IBM processors (:func:`heavy_hex_edges`): the couplings
    are split into sets of disjoint pairs (:func:`edge_matchings`; 2 for a
    line, 3 for the heavy-hex graph) and layer ``l`` applies the set
    ``l mod (number of sets)``. ``RZZ(theta) = exp(-i theta Z (x) Z / 2)`` is a
    native (fractional) gate of IBM Heron processors, or two CZ gates
    otherwise; ``RZ`` is virtual (free) and ``RY`` costs one native rotation.
    For a line of qubits this is the PCE brickwork with ``RZZ`` in place of
    the MS gate, which is locally equivalent to it.

    Returns:
        ``(gates, n_params)``.
    """
    matchings = edge_matchings(heavy_hex_edges(n))
    gates, p = [], 0
    for layer in range(layers):
        # Qubit q gets RY(params[p + 2q]) then RZ(params[p + 2q + 1]). All RY gates are listed before
        # all RZ gates (gates on different qubits commute), so the RZ and RZZ gates of the layer form
        # one run of diagonal gates, which the simulator applies at once.
        gates += [Gate("R", "Y", (q,), (p + 2 * q,)) for q in range(n)]
        gates += [Gate("R", "Z", (q,), (p + 2 * q + 1,)) for q in range(n)]
        p += 2 * n
        for a, b in (matchings[layer % len(matchings)] if matchings else []):
            gates.append(Gate("ZZ", "", (a, b), (p,)))
            p += 1
    return gates, p


#: Available ansatz architectures.
ANSATZE = {"brickwork": brickwork, "heavy_hex": heavy_hex}


def build_ansatz(ansatz: str, n: int, layers: int) -> tuple[list[Gate], int]:
    """Gates and number of parameters of an ansatz (``"brickwork"`` or ``"heavy_hex"``)."""
    if ansatz not in ANSATZE:
        raise ValueError(f"unknown ansatz {ansatz!r}; choose from {sorted(ANSATZE)}")
    return ANSATZE[ansatz](n, layers)


def default_layers(n: int, m: int, ansatz: str = "brickwork") -> int:
    """Fewest layers with at least ``m`` parameters (the PCE paper's rule)."""
    layers = 1
    while build_ansatz(ansatz, n, layers)[1] < m:
        layers += 1
    return layers


def _rotations_batch(P: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``exp(-i theta P / 2)`` and its derivative for a batch of Paulis ``(g, 2, 2)`` and angles ``(g,)``."""
    c, s = np.cos(theta / 2)[:, None, None], np.sin(theta / 2)[:, None, None]
    return c * _I2 - 1j * s * P, -0.5 * s * _I2 - 0.5j * c * P


def _ms_batch(phi0: np.ndarray, phi1: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MS gates ``(g, 4, 4)`` and their derivatives ``(g, 3, 4, 4)`` with respect to ``phi0``, ``phi1``, ``theta``.

    ``exp(-i theta/2 sigma_phi0 (x) sigma_phi1)`` has ``cos(theta/2)`` on the
    diagonal and ``-i sin(theta/2)`` times the phases ``p = exp(-i(phi0 + phi1))``
    (entries 03 and, conjugated, 30) and ``q = exp(-i(phi0 - phi1))`` (12 and 21).
    """
    g = len(theta)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    p, q = np.exp(-1j * (phi0 + phi1)), np.exp(-1j * (phi0 - phi1))
    anti = np.zeros((g, 4, 4), dtype=complex)                    # off-diagonal pattern of sigma (x) sigma
    anti[:, 0, 3], anti[:, 3, 0], anti[:, 1, 2], anti[:, 2, 1] = p, p.conj(), q, q.conj()
    eye = np.eye(4)
    U = c[:, None, None] * eye - 1j * s[:, None, None] * anti
    dU = np.zeros((g, 3, 4, 4), dtype=complex)
    # d/dphi0 multiplies p and q by -i (their conjugates by +i); d/dphi1 does the same to p and the opposite to q.
    dU[:, 0, 0, 3], dU[:, 0, 3, 0], dU[:, 0, 1, 2], dU[:, 0, 2, 1] = -s * p, s * p.conj(), -s * q, s * q.conj()
    dU[:, 1, 0, 3], dU[:, 1, 3, 0], dU[:, 1, 1, 2], dU[:, 1, 2, 1] = -s * p, s * p.conj(), s * q, -s * q.conj()
    dU[:, 2] = -0.5 * s[:, None, None] * eye - 0.5j * c[:, None, None] * anti
    return U, dU


def ms_matrix(phi0: float, phi1: float, theta: float) -> np.ndarray:
    """Molmer-Sorensen gate ``exp(-i theta/2 sigma_phi0 (x) sigma_phi1)`` (first qubit most significant)."""
    return _ms_batch(np.array([phi0]), np.array([phi1]), np.array([theta]))[0][0]


def _rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def ms_composed(a0: float, a1: float, theta: float, b0: float, b1: float) -> np.ndarray:
    """``(RZ(a0) x RZ(a1)) exp(-i theta/2 X x X) (RZ(b0) x RZ(b1))``.

    With ``a = (phi0, phi1)`` and ``b = (-phi0, -phi1)`` this is
    :func:`ms_matrix`; shifting one of the four ``RZ`` angles separately gives
    the terms of the parameter-shift rule of ``phi0`` and ``phi1``.
    """
    xx = np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * np.kron(_PAULI["X"], _PAULI["X"])
    return np.kron(_rz(a0), _rz(a1)) @ xx @ np.kron(_rz(b0), _rz(b1))


def _apply(psi: np.ndarray, gate: Gate, U: np.ndarray) -> np.ndarray:
    """Apply a gate to a flat state vector (qubit 0 is the most significant bit).

    ``ZZ`` gates are diagonal and ``U`` is then their diagonal (``2**n``
    phases). For the other gates, whose wires are consecutive, the state is
    viewed as an array of shape ``(2**first_wire, 2**width, rest)`` and
    multiplied by the gate matrix along the middle axis.
    """
    if gate.kind == "ZZ":
        return psi * U
    return _matmul(psi, U, gate.wires[0])


def _matmul(psi: np.ndarray, U: np.ndarray, first: int) -> np.ndarray:
    """Multiply the consecutive qubits ``first, first + 1, ...`` (as many as ``U`` acts on) of a flat state by ``U``.

    The state is viewed as ``(2**first, width, rest)``. In large states, when
    ``rest`` is small (gates on the last qubits), the many tiny products along
    the middle axis are slow, and one matrix product with ``U (x) I_rest`` is
    used instead (in small states, building ``U (x) I_rest`` costs more).
    """
    width = U.shape[0]
    rest = psi.size // (width << first)
    if rest <= 8 and psi.size >= 4096:
        return (psi.reshape(-1, width * rest) @ np.kron(U, np.eye(rest)).T).reshape(-1)
    return (U @ psi.reshape(1 << first, width, rest)).reshape(-1)


def _rotate_qubits(psi: np.ndarray, V: np.ndarray, n: int) -> np.ndarray:
    """Apply the single-qubit matrix ``V`` to every qubit of a flat state vector."""
    for q in range(n):
        psi = _matmul(psi, V, q)
    return psi


# --------------------------------------------------------------------------
# Correlator readout
# --------------------------------------------------------------------------

class _Readout:
    """Correlators of the Pauli strings from the outcome distributions of the measurement settings.

    For one setting (all qubits measured in the basis of ``axis``), the
    correlator of a string on qubits ``Q`` is ``sum_z p(z) prod_{q in Q} s_q(z)``
    with ``s_q(z) = (-1)^z_q``. Single-qubit (``k = 1``) and two-qubit
    (``k = 2``) strings are read from the matrix ``S`` of single-qubit signs
    (``2^n x n``); longer strings use one column of signs per string.
    """

    def __init__(self, n: int, strings: list[tuple[str, tuple[int, ...]]]):
        self.n, self.m = n, len(strings)
        self.k = len(strings[0][1])
        self.settings = [axis for axis in AXES if any(a == axis for a, _ in strings)]
        z = (np.arange(1 << n)[:, None] >> (n - 1 - np.arange(n))) & 1        # qubit 0 = most significant bit
        self.signs = 1.0 - 2.0 * z                                               # (2^n, n)
        self.groups = {}
        for axis in self.settings:
            pos = np.array([i for i, (a, _) in enumerate(strings) if a == axis])
            qubits = np.array([strings[i][1] for i in pos])                       # (m_axis, k)
            dense = None
            if self.k > 2:
                dense = np.prod(self.signs[:, qubits], axis=2)                    # (2^n, m_axis)
            self.groups[axis] = (pos, qubits, dense)

    def correlators(self, probs: dict[str, np.ndarray]) -> np.ndarray:
        """Correlators of all strings from the distribution of every setting."""
        c = np.empty(self.m)
        for axis, (pos, qubits, dense) in self.groups.items():
            p = probs[axis]
            if self.k == 1:
                c[pos] = (p @ self.signs)[qubits[:, 0]]
            elif self.k == 2:
                gram = (self.signs * p[:, None]).T @ self.signs                  # (n, n)
                c[pos] = gram[qubits[:, 0], qubits[:, 1]]
            else:
                c[pos] = p @ dense
        return c

    def diagonal(self, axis: str, w: np.ndarray) -> np.ndarray:
        """Diagonal (in the basis of ``axis``) of ``O = sum_i w_i Pi_i`` over the strings of that axis."""
        pos, qubits, dense = self.groups[axis]
        if self.k == 1:
            weights = np.zeros(self.n)
            np.add.at(weights, qubits[:, 0], w[pos])
            return self.signs @ weights
        if self.k == 2:
            W = np.zeros((self.n, self.n))
            W[qubits[:, 0], qubits[:, 1]] = w[pos]
            W = W + W.T
            return 0.5 * np.einsum("zi,zi->z", self.signs @ W, self.signs)
        return dense @ w[pos]


# --------------------------------------------------------------------------
# Circuit simulation
# --------------------------------------------------------------------------

class StateVectorCircuit:
    """Exact numpy simulation of the ansatz and of the Pauli-correlation readout.

    Args:
        n: Number of qubits.
        layers: Number of ansatz layers.
        strings: Pauli strings of the encoding (see :func:`pauli_strings`).
        ansatz: ``"brickwork"`` (the PCE paper's circuit, with MS gates) or
            ``"heavy_hex"`` (IBM-native, see :func:`heavy_hex`).
    """

    def __init__(self, n: int, layers: int, strings: list[tuple[str, tuple[int, ...]]], ansatz: str = "brickwork"):
        self.n, self.layers, self.strings, self.ansatz = n, layers, strings, ansatz
        self.gates, self.n_params = build_ansatz(ansatz, n, layers)
        self.readout = _Readout(n, strings)
        #: Axes measured (one measurement setting each).
        self.settings = self.readout.settings
        # Gate matrices are built for all gates of a kind at once (see _matrices).
        self._rot_idx = [i for i, g in enumerate(self.gates) if g.kind == "R"]
        self._rot_param = np.array([self.gates[i].params[0] for i in self._rot_idx], dtype=int)
        self._rot_pauli = np.array([_PAULI[self.gates[i].axis] for i in self._rot_idx]).reshape(-1, 2, 2)
        self._ms_idx = [i for i, g in enumerate(self.gates) if g.kind == "MS"]
        self._ms_param = np.array([self.gates[i].params for i in self._ms_idx], dtype=int).reshape(-1, 3)
        # ZZ gates: whether the two qubits have equal bits in every basis state (phase e^{-i theta/2} if so).
        self._zz_idx = [i for i, g in enumerate(self.gates) if g.kind == "ZZ"]
        self._zz_param = np.array([self.gates[i].params[0] for i in self._zz_idx], dtype=int)
        bits = (np.arange(1 << n)[:, None] >> (n - 1 - np.arange(n))) & 1
        self._zz_same = None
        if self._zz_idx:
            self._zz_same = np.array([bits[:, self.gates[i].wires[0]] == bits[:, self.gates[i].wires[1]]
                                      for i in self._zz_idx])
        # Simulation steps: single gates (their index) and runs of consecutive diagonal gates (Z rotations
        # and ZZ gates), as (parameter indices, rows of the sign table). The gates of a run commute, so a
        # run is applied as one vector of phases, and the gradient of all its parameters takes one product
        # with the sign table (see value_and_grad).
        self._steps, rows = [], {}
        for i, g in enumerate(self.gates):
            if g.kind == "ZZ" or (g.kind == "R" and g.axis == "Z"):
                if not self._steps or isinstance(self._steps[-1], int):
                    self._steps.append(([], []))
                self._steps[-1][0].append(g.params[0])
                self._steps[-1][1].append(rows.setdefault(g.wires, len(rows)))
            else:
                self._steps.append(i)
        self._steps = [s if isinstance(s, int) else (np.array(s[0]), np.array(s[1])) for s in self._steps]
        #: Diagonal (+-1 per basis state) of every generator of a diagonal gate: Z of one qubit or Z (x) Z of two.
        self._signs = np.array([np.prod(1 - 2 * bits[:, list(w)], axis=1) for w in rows], dtype=float)

    @property
    def m(self) -> int:
        """Number of encoded variables."""
        return len(self.strings)

    def _matrices(self, params: np.ndarray, diagonal: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
        """Matrix of every gate and its derivatives with respect to the gate's parameters, in circuit order.

        With ``diagonal=False`` the ZZ gates (applied in runs, see
        :meth:`_run`) are left as ``None``.
        """
        out: list = [None] * len(self.gates)
        if self._rot_idx:
            U, dU = _rotations_batch(self._rot_pauli, params[self._rot_param])
            for j, i in enumerate(self._rot_idx):
                out[i] = (U[j], dU[j:j + 1])
        if self._ms_idx:
            p = params[self._ms_param]
            U, dU = _ms_batch(p[:, 0], p[:, 1], p[:, 2])
            for j, i in enumerate(self._ms_idx):
                out[i] = (U[j], dU[j])
        for j, i in enumerate(self._zz_idx if diagonal else []):
            U, dU = self._zz_diagonal(j, params[self._zz_param[j]])
            out[i] = (U, dU[None, :])
        return out

    def _run(self, params: np.ndarray) -> tuple[np.ndarray, list]:
        """Output state, applying every run of diagonal gates at once.

        Returns:
            ``(state, ops)``: ``ops`` has, per step, the gate's matrix and
            derivatives, or the phases ``exp(-i/2 sum_k theta_k s_k)`` of a
            run (``s_k``: rows of the sign table).
        """
        mats = self._matrices(params, diagonal=False)
        psi = np.zeros(1 << self.n, dtype=complex)
        psi[0] = 1.0
        ops = []
        for step in self._steps:
            if isinstance(step, int):
                ops.append(mats[step])
                psi = _apply(psi, self.gates[step], mats[step][0])
            else:
                ps, rows = step
                angles = np.bincount(rows, weights=params[ps], minlength=len(self._signs)) @ self._signs
                ops.append(np.exp(-0.5j * angles))
                psi = psi * ops[-1]
        return psi, ops

    def _zz_diagonal(self, j: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Diagonal of the ``j``-th ZZ gate and of its derivative with respect to ``theta``."""
        a = np.exp(-0.5j * theta)
        same = self._zz_same[j]
        return np.where(same, a, a.conjugate()), np.where(same, -0.5j * a, 0.5j * a.conjugate())

    def _forward(self, matrices: list[np.ndarray]) -> np.ndarray:
        psi = np.zeros(1 << self.n, dtype=complex)
        psi[0] = 1.0
        for g, U in zip(self.gates, matrices):
            psi = _apply(psi, g, U)
        return psi

    def state(self, params: np.ndarray) -> np.ndarray:
        """Output state, as a flat vector of ``2**n`` amplitudes (qubit 0 is the most significant bit)."""
        return self._run(params)[0]

    def _rotated(self, psi: np.ndarray) -> dict[str, np.ndarray]:
        """The state after the basis change of every measurement setting."""
        return {a: psi if a == "Z" else _rotate_qubits(psi, _BASIS[a], self.n) for a in self.settings}

    def _probabilities(self, psi: np.ndarray) -> dict[str, np.ndarray]:
        return {a: np.abs(v) ** 2 for a, v in self._rotated(psi).items()}

    def probabilities(self, params: np.ndarray) -> dict[str, np.ndarray]:
        """Outcome probabilities of every measurement setting (``2**n`` each)."""
        return self._probabilities(self.state(params))

    def correlators_from_probabilities(self, probs: dict[str, np.ndarray]) -> np.ndarray:
        """Correlators ``<Pi_i>`` from the outcome distribution of every setting."""
        return self.readout.correlators(probs)

    def correlators(self, params: np.ndarray) -> np.ndarray:
        """Exact correlators ``<Pi_i>`` for the parameters ``params``."""
        return self.readout.correlators(self.probabilities(params))

    def sample(self, probs: dict[str, np.ndarray], shots: int | None, rng: np.random.Generator) -> np.ndarray:
        """Correlators estimated from ``shots`` measurements per setting (exact if ``shots`` is ``None``)."""
        if shots is None:
            return self.readout.correlators(probs)
        freq = {a: rng.multinomial(shots, p / p.sum()) / shots for a, p in probs.items()}
        return self.readout.correlators(freq)

    def estimate(self, params: np.ndarray, shots: int | None, rng: np.random.Generator) -> np.ndarray:
        """Correlators of the circuit with parameters ``params``, from ``shots`` per setting."""
        return self.sample(self.probabilities(params), shots, rng)

    def value_and_grad(self, params: np.ndarray, loss) -> tuple[float, np.ndarray, np.ndarray]:
        """Loss of the exact correlators and its exact gradient (adjoint method).

        Args:
            params: Circuit parameters.
            loss: Callable ``c -> (value, dvalue/dc)``.

        Returns:
            ``(value, gradient, correlators)``.
        """
        psi, ops = self._run(params)
        rotated = self._rotated(psi)
        c = self.readout.correlators({a: np.abs(v) ** 2 for a, v in rotated.items()})
        value, w = loss(c)
        # Adjoint state lam = O psi with O = sum_i w_i Pi_i, built in each measurement basis.
        lam = np.zeros_like(psi)
        for axis in self.settings:
            v = rotated[axis] * self.readout.diagonal(axis, w)
            lam += v if axis == "Z" else _rotate_qubits(v, _BASIS[axis].conj().T, self.n)
        grad = np.zeros(self.n_params)
        for step, op in zip(reversed(self._steps), reversed(ops)):
            if isinstance(step, int):
                g, (U, dUs) = self.gates[step], op
                Ud = U.conj().T
                psi = _apply(psi, g, Ud)                    # state before the gate
                for p, dU in zip(g.params, dUs):
                    grad[p] = 2.0 * np.real(np.vdot(lam, _apply(psi, g, dU)))
                lam = _apply(lam, g, Ud)
                continue
            # Run of diagonal gates exp(-i theta_k s_k / 2): the derivative inserts -i s_k / 2, so
            # d<O>/d(theta_k) = Im <lam| s_k |psi>, with psi and lam taken after the run (its gates commute).
            ps, rows = step
            grad[ps] = (self._signs @ (lam.conj() * psi).imag)[rows]
            back = op.conj()
            psi, lam = psi * back, lam * back
        return value, grad, c

    def shift_rules(self, params: np.ndarray) -> list[list[tuple[int, np.ndarray, float]]]:
        """Parameter-shift rule of every parameter, as ``(gate index, replacement gate matrix, coefficient)`` terms.

        ``d<O>/d(param) = sum_terms coefficient * <O>(circuit with the gate replaced)``.
        Rotations, ZZ gates and the ``theta`` of MS gates have generators with
        eigenvalues ``+-1/2`` (two terms, shifts of ``+-pi/2``). ``phi0`` and
        ``phi1`` enter an MS gate through two ``RZ`` rotations
        (:func:`ms_composed`), so their rules have four terms.
        """
        return [terms for _, terms in sorted(self._iter_shift_rules(params), key=lambda pt: pt[0])]

    def _iter_shift_rules(self, params: np.ndarray):
        """Yield ``(parameter index, terms)`` of the parameter-shift rule of every parameter, one gate at a time."""
        zz = {i: j for j, i in enumerate(self._zz_idx)}
        for gi, g in enumerate(self.gates):
            if g.kind == "R":
                (p,) = g.params
                U = _rotations_batch(_PAULI[g.axis][None], np.array([params[p] + HALF_PI, params[p] - HALF_PI]))[0]
                yield p, [(gi, U[0], 0.5), (gi, U[1], -0.5)]
                continue
            if g.kind == "ZZ":
                (p,) = g.params
                yield p, [(gi, self._zz_diagonal(zz[gi], params[p] + HALF_PI)[0], 0.5),
                          (gi, self._zz_diagonal(zz[gi], params[p] - HALF_PI)[0], -0.5)]
                continue
            p0, p1, pt = g.params
            f0, f1, th = params[p0], params[p1], params[pt]
            yield p0, [(gi, ms_composed(f0 + HALF_PI, f1, th, -f0, -f1), 0.5),
                       (gi, ms_composed(f0 - HALF_PI, f1, th, -f0, -f1), -0.5),
                       (gi, ms_composed(f0, f1, th, -f0 + HALF_PI, -f1), -0.5),
                       (gi, ms_composed(f0, f1, th, -f0 - HALF_PI, -f1), 0.5)]
            yield p1, [(gi, ms_composed(f0, f1 + HALF_PI, th, -f0, -f1), 0.5),
                       (gi, ms_composed(f0, f1 - HALF_PI, th, -f0, -f1), -0.5),
                       (gi, ms_composed(f0, f1, th, -f0, -f1 + HALF_PI), -0.5),
                       (gi, ms_composed(f0, f1, th, -f0, -f1 - HALF_PI), 0.5)]
            yield pt, [(gi, ms_matrix(f0, f1, th + HALF_PI), 0.5), (gi, ms_matrix(f0, f1, th - HALF_PI), -0.5)]

    def shifted_jacobian(self, params: np.ndarray, shots: int | None,
                         rng: np.random.Generator) -> tuple[np.ndarray, int]:
        """Jacobian ``d<Pi_i>/d(param)`` from the parameter-shift rule, estimated with ``shots`` per setting.

        Returns:
            ``(jacobian of shape (m, n_params), circuit executions)``.
        """
        base = [U for U, _ in self._matrices(params)]
        jac = np.zeros((self.m, self.n_params))
        executions = 0
        for p, terms in self._iter_shift_rules(params):
            for gi, U, coef in terms:
                mats = list(base)
                mats[gi] = U
                jac[:, p] += coef * self.sample(self._probabilities(self._forward(mats)), shots, rng)
                executions += 1
        return jac, executions


# --------------------------------------------------------------------------
# PennyLane version of the circuit (verification of the numpy simulator)
# --------------------------------------------------------------------------

def _pennylane_ansatz(circuit: StateVectorCircuit, params) -> None:
    """Queue the ansatz in a PennyLane tape.

    MS gates are compiled as ``RZ(phi) IsingXX(theta) RZ(-phi)`` on each qubit
    and ZZ gates are ``IsingZZ``.
    """
    import pennylane as qml

    rot = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}
    for g in circuit.gates:
        if g.kind == "R":
            rot[g.axis](params[g.params[0]], wires=g.wires[0])
            continue
        a, b = g.wires
        if g.kind == "ZZ":
            qml.IsingZZ(params[g.params[0]], wires=[a, b])
            continue
        phi0, phi1, theta = (params[i] for i in g.params)
        qml.RZ(-phi0, wires=a)
        qml.RZ(-phi1, wires=b)
        qml.IsingXX(theta, wires=[a, b])
        qml.RZ(phi0, wires=a)
        qml.RZ(phi1, wires=b)


def pennylane_correlators(circuit: StateVectorCircuit, device: str = "default.qubit"):
    """PennyLane QNode ``params -> (<Pi_1>, ..., <Pi_m>)`` of the same circuit.

    Args:
        circuit: Circuit to rebuild.
        device: PennyLane device.
    """
    import pennylane as qml

    pauli = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    dev = qml.device(device, wires=circuit.n)

    @qml.qnode(dev)
    def qnode(params):
        _pennylane_ansatz(circuit, params)
        return [qml.expval(qml.prod(*(pauli[axis](q) for q in qubits))) for axis, qubits in circuit.strings]

    return qnode


def pennylane_probabilities(circuit: StateVectorCircuit, device: str = "default.qubit"):
    """Function ``params -> {axis: outcome probabilities}`` of every measurement setting, in PennyLane."""
    import pennylane as qml

    dev = qml.device(device, wires=circuit.n)

    @qml.qnode(dev)
    def qnode(params, axis):
        _pennylane_ansatz(circuit, params)
        for q in range(circuit.n):
            if axis == "X":
                qml.Hadamard(wires=q)
            elif axis == "Y":
                qml.adjoint(qml.S(wires=q))
                qml.Hadamard(wires=q)
        return qml.probs(wires=range(circuit.n))

    return lambda params: {axis: np.asarray(qnode(params, axis), dtype=float) for axis in circuit.settings}


def verify_against_pennylane(m: int = 30, k: int = 2, layers: int = 3, seed: int = 0,
                             device: str = "default.qubit", ansatz: str = "brickwork") -> dict[str, float]:
    """Compare the numpy simulator with PennyLane on random parameters.

    Returns the largest absolute differences of the correlators, of the
    outcome probabilities, of the gradient of a random weighted sum of
    correlators (PennyLane's Jacobian, contracted with the same weights) and
    of the same gradient obtained from the exact parameter-shift rule.
    """
    import pennylane as qml
    from pennylane import numpy as pnp

    n = min_qubits(m, k)
    circuit = StateVectorCircuit(n, layers, pauli_strings(m, k, n), ansatz)
    rng = np.random.default_rng(seed)
    params = rng.uniform(-np.pi, np.pi, circuit.n_params)
    w = rng.normal(size=m)
    _, grad, c = circuit.value_and_grad(params, lambda c: (float(w @ c), w))
    qnode = pennylane_correlators(circuit, device)
    c_pl = np.array(qnode(params), dtype=float)
    jac = qml.jacobian(lambda p: qml.math.stack(qnode(p)))(pnp.array(params, requires_grad=True))
    probs, probs_pl = circuit.probabilities(params), pennylane_probabilities(circuit, device)(params)
    shift_jac, _ = circuit.shifted_jacobian(params, None, rng)
    return {"correlators": float(np.max(np.abs(c - c_pl))),
            "probabilities": max(float(np.max(np.abs(probs[a] - probs_pl[a]))) for a in circuit.settings),
            "gradient": float(np.max(np.abs(grad - np.asarray(jac).T @ w))),
            "parameter_shift": float(np.max(np.abs(grad - shift_jac.T @ w)))}


# --------------------------------------------------------------------------
# Configuration and results
# --------------------------------------------------------------------------

@dataclass
class PCEConfig:
    """Hyper-parameters of the PCE solver.

    Attributes:
        optimizer: ``"adam"``, ``"spsa"``, ``"slsqp"`` or ``"exact"`` (no
            circuit: the reference solution, which must come from an exact
            enumeration).
        training_shots: Shots per measurement setting used to estimate the
            loss (and, for Adam and SLSQP, the parameter-shift gradient)
            during training; ``None`` trains on the exact simulation.
        k: Order of the Pauli strings (compression degree; ``m = O(n^k)``).
        n_qubits: Number of qubits (``None`` = the fewest that fit the variables).
        ansatz: Circuit architecture: ``"brickwork"`` (the PCE paper's, with
            MS gates between neighbors) or ``"heavy_hex"`` (IBM-native: ``RY``,
            ``RZ`` and ``RZZ`` on the heavy-hex coupling graph).
        layers: Ansatz layers (``None`` = the fewest with at least ``m`` parameters).
        alpha: Scale of the correlators inside ``tanh`` (``None`` = ``1.5 n^floor(k/2)``).
        beta: Weight of the regularization term (the paper's ``1/2``).
        cardinality: Relaxation of the constrained blocks in the loss:
            ``"offset"`` (soft top-``w`` with a per-block offset; exact
            cardinality, no penalty) or ``"penalty"`` (quadratic penalty).
            See the module docstring.
        penalty_scale: Multiplier of the cardinality penalty weight (``1`` =
            the bound of :func:`penalty_weight`; ``"penalty"`` only).
        learning_rate: Adam step size.
        max_steps: Largest number of Adam steps or SLSQP iterations.
        patience: Exact Adam stops when the loss improved by less than
            ``tolerance`` over the last ``patience`` steps (the PCE paper uses
            50 steps); shot-based Adam runs ``max_steps`` steps.
        tolerance: See ``patience`` (relative, since the loss is divided by ``nu``).
        spsa_iterations: Number of SPSA iterations (two loss evaluations each).
        spsa_c: Initial SPSA perturbation (radians).
        spsa_target_step: Size of the first SPSA update (radians); the gain
            ``a`` is calibrated from ``spsa_calibration`` gradient samples so
            that the first step has this size.
        spsa_alpha, spsa_gamma: Spall's decay exponents of the gain and of the
            perturbation.
        spsa_calibration: Gradient samples used to calibrate the SPSA gain.
        readout_shots: Shots per setting of the final readout (``None`` = exact).
        search_rounds: Rounds of local search after decoding (the paper uses one).
        log_every: Record the exact loss every ``log_every`` iterations.
        expected_steps: Expected optimizer iterations, used only to report
            progress (``None`` = ``max_steps`` or ``spsa_iterations``).
    """

    optimizer: str = "adam"
    training_shots: int | None = None
    k: int = 2
    n_qubits: int | None = None
    ansatz: str = "brickwork"
    layers: int | None = 6
    alpha: float | None = 3.0
    beta: float = 0.5
    cardinality: str = "offset"
    penalty_scale: float = 1.0
    learning_rate: float = 0.01
    max_steps: int = 20000
    patience: int = 50
    tolerance: float = 1e-6
    spsa_iterations: int = 10000
    spsa_c: float = 0.1
    spsa_target_step: float = 0.2
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    spsa_calibration: int = 10
    readout_shots: int | None = 1024
    search_rounds: int = 1
    log_every: int = 10
    expected_steps: int | None = None


@dataclass
class PCEResult:
    """Outcome of one PCE run.

    Quality fields (``*_gap``, ``*_rank``, ``*_ratio``) score an energy
    against the run's :class:`qopt.feasible.Reference`: the normalized gap is
    0 for the reference and 1 for an average random feasible bitstring; rank
    and approximation ratio are ``nan`` unless the reference is exact.

    Attributes:
        bits, energy: Answer (decoded bitstring after the local search) and its energy.
        raw_bits, raw_energy: Decoded bitstring before the search and its energy.
        sign_feasible: Whether plain sign decoding already satisfied every constraint.
        random_energy: Random feasible bitstring followed by the local search.
        initial_raw_energy, initial_energy: Decoded untrained circuit, alone and after the search.
        greedy_energy, greedy_evaluations: Steepest-descent local search.
        reference: The :class:`qopt.feasible.Reference` used for scoring.
        n_qubits, layers, n_params: Size of the circuit.
        loss: Exact loss of the returned parameters.
        steps: Optimizer iterations.
        loss_evaluations: Loss evaluations (exact or estimated) during training.
        circuit_executions: Circuit runs during training (each one measures
            every setting); parameter-shift gradients count one run per term.
        training_shots_used, readout_shots_used: Shots spent (all settings).
        search_evaluations: Candidate moves evaluated by the final local search.
        correlator_mean_abs: Mean ``|<Pi_i>|`` of the returned parameters.
        penalty: Cardinality penalty weight ``rho``.
        stop_reason: Why training stopped.
        seconds: Wall-clock time of the run.
        params: Returned circuit parameters (empty for ``optimizer="exact"``).
        history: Exact loss recorded during training
            (``{"iteration", "evaluations", "loss", "loss_estimate"}``).
    """

    bits: np.ndarray
    energy: float
    raw_bits: np.ndarray
    raw_energy: float
    sign_feasible: bool
    random_energy: float
    initial_raw_energy: float
    initial_energy: float
    greedy_energy: float
    greedy_evaluations: int
    reference: Reference
    n_qubits: int
    layers: int
    n_params: int
    loss: float
    steps: int
    loss_evaluations: int
    circuit_executions: int
    training_shots_used: int
    readout_shots_used: int
    search_evaluations: int
    correlator_mean_abs: float
    penalty: float
    stop_reason: str
    seconds: float
    params: np.ndarray = field(default_factory=lambda: np.zeros(0))
    history: list[dict] = field(default_factory=list)

    def quality(self) -> dict[str, float]:
        """Gap, rank and approximation ratio of the answer, of the decoded bitstring and of the baselines."""
        out = {"reference_kind": self.reference.kind, "reference_energy": self.reference.energy,
               "n_feasible": float(self.reference.n_feasible)}
        for name, e in (("answer", self.energy), ("raw", self.raw_energy), ("random", self.random_energy),
                        ("initial_raw", self.initial_raw_energy), ("initial", self.initial_energy),
                        ("greedy", self.greedy_energy)):
            out[f"{name}_energy"] = e
            out[f"{name}_gap"] = self.reference.gap(e) if np.isfinite(e) else np.nan
            out[f"{name}_rank"] = self.reference.rank(e) if np.isfinite(e) else np.nan
            out[f"{name}_ratio"] = self.reference.ratio(e) if np.isfinite(e) else np.nan
            out[f"{name}_is_best"] = bool(np.isfinite(e) and self.reference.is_best(e))
        return out

    def summary(self) -> dict:
        """Flat dictionary of the scalar fields (for CSV logging)."""
        keys = ("energy", "raw_energy", "sign_feasible", "greedy_evaluations", "n_qubits", "layers", "n_params",
                "loss", "steps", "loss_evaluations", "circuit_executions", "training_shots_used",
                "readout_shots_used", "search_evaluations", "correlator_mean_abs", "penalty", "stop_reason",
                "seconds")
        return {**{k: getattr(self, k) for k in keys}, **self.quality()}


# --------------------------------------------------------------------------
# Problem: encoding, penalties and loss
# --------------------------------------------------------------------------

def penalty_weight(qubo: QUBO, blocks: list[Block]) -> float:
    """Penalty weight ``rho`` of the cardinality constraints.

    It is the largest change of the objective caused by flipping one variable
    of a feasible bitstring, ``max_i (|lin_i| + sum of the W largest |quad_ij|)``,
    with ``W`` the largest number of ones of a feasible bitstring. With this
    weight, no single flip that leaves the feasible set lowers the penalized
    objective.
    """
    W = sum(b.weight if b.constrained else b.size for b in blocks)
    A = np.abs(np.asarray(qubo.quad, dtype=float))
    top = -np.sort(-A, axis=1)[:, :W].sum(axis=1) if W else np.zeros(qubo.n)
    return float(np.max(np.abs(qubo.lin) + top))


def mean_spin_bound(n: int, strings: list[tuple[str, tuple[int, ...]]], alpha: float) -> float:
    """Lowest mean of ``tanh(alpha <Pi_i>)`` that the two-body correlators of one block can reach.

    For each Pauli axis ``P``, ``sum_{a<b} <P_a P_b> = (<(sum_a P_a)^2> - n) / 2 >= -n/2``.
    The correlators of that axis outside the block are at most 1 each, and so
    are the ``p`` positive correlators of the block, so its ``m - p`` negative
    correlators add up, in absolute value, to at most ``Gamma + p`` with
    ``Gamma = sum_P [n/2 + C(n, 2) - e_P]`` (``e_P``: strings of axis ``P`` in
    the block). As ``tanh`` is concave on ``[0, inf)``, their mean ``tanh`` is
    at least ``-tanh(alpha (Gamma + p) / (m - p))``; the bound is the minimum
    over ``p``.

    The cardinality penalty asks the block for a mean of ``2w/m - 1``; when
    that is below this bound (sparse choices: 320 variables on 16 qubits,
    ``alpha = 3``, give -0.725, while ``w = 32`` asks for -0.8), no circuit can
    satisfy it, which is why the offset relaxation is used instead.

    Args:
        n: Qubits.
        strings: Two-body Pauli strings of the block (see :func:`pauli_strings`).
        alpha: Scale of the correlators inside ``tanh``.
    """
    if any(len(q) != 2 for _, q in strings):
        raise ValueError("the bound holds for two-body strings (k = 2)")
    m = len(strings)
    counts = {axis: sum(a == axis for a, _ in strings) for axis in {a for a, _ in strings}}
    gamma = sum(n / 2 + comb(n, 2) - e for e in counts.values())
    p = np.arange(m)
    return float(np.min(-(m - p) * np.tanh(alpha * (gamma + p) / (m - p))) / m)


def resolve(m: int, config: PCEConfig) -> tuple[int, int, float]:
    """Number of qubits, layers and ``alpha`` actually used for ``m`` variables."""
    n = config.n_qubits or min_qubits(m, config.k)
    layers = config.layers or default_layers(n, m, config.ansatz)
    alpha = config.alpha if config.alpha is not None else 1.5 * n ** (config.k // 2)
    return n, layers, alpha


class PCEProblem:
    """A block-constrained QUBO encoded in Pauli correlations, with the relaxed loss.

    Args:
        qubo: QUBO to minimize.
        blocks: Block layout (constrained and free blocks).
        config: Solver hyper-parameters.
    """

    def __init__(self, qubo: QUBO, blocks: list[Block], config: PCEConfig):
        if config.cardinality not in ("offset", "penalty"):
            raise ValueError(f"unknown cardinality relaxation {config.cardinality!r}; use 'offset' or 'penalty'")
        self.qubo, self.blocks, self.config = qubo, blocks, config
        m = qubo.n
        self.n_qubits, self.layers, self.alpha = resolve(m, config)
        self.circuit = StateVectorCircuit(self.n_qubits, self.layers, pauli_strings(m, config.k, self.n_qubits),
                                          config.ansatz)
        self.rho = config.penalty_scale * penalty_weight(qubo, blocks) if config.cardinality == "penalty" else 0.0
        pairs = [(s, b) for s, b in zip(block_slices(blocks), blocks) if b.constrained]
        self.slices = [s for s, _ in pairs]
        self.weights = [b.weight for _, b in pairs]
        self.targets = np.array([2 * b.weight - b.size for _, b in pairs], dtype=float)
        self.h, self.J, _ = qubo.ising()
        nu = 0.5 * (np.abs(self.h).sum() + 0.5 * np.abs(self.J).sum())
        self.nu = nu if nu > 1e-12 else 1.0

    def relax(self, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
        """Relaxed spins ``t~ = 2 x~ - 1`` of the correlators and what their gradient needs.

        Returns:
            ``(t~, dt~/dz, offsets)``: the relaxed spins, the derivative of
            each one with respect to its own shifted argument
            ``z_i = alpha (c_i - b)`` times ``alpha``, and, for every
            constrained block in ``"offset"`` mode, its slice (used to project
            the gradient on the constraint).
        """
        t = np.tanh(self.alpha * c)
        tt, dt = t.copy(), self.alpha * (1.0 - t * t)
        blocks = []
        if self.config.cardinality == "offset":
            for s, w in zip(self.slices, self.weights):
                cb = c[s]
                if w in (0, len(cb)):                       # nothing to choose: fixed 0s or 1s
                    tt[s], dt[s] = (1.0 if w else -1.0), 0.0
                    continue
                # Offset b with sum_i (1 + tanh(alpha (c_i - b))) / 2 = w (the sum decreases with b).
                lo, hi = float(cb.min()) - 20.0 / self.alpha, float(cb.max()) + 20.0 / self.alpha
                for _ in range(60):
                    b = 0.5 * (lo + hi)
                    if 0.5 * np.sum(1.0 + np.tanh(self.alpha * (cb - b))) > w:
                        lo = b
                    else:
                        hi = b
                tb = np.tanh(self.alpha * (cb - 0.5 * (lo + hi)))
                tt[s], dt[s] = tb, self.alpha * (1.0 - tb * tb)
                blocks.append(s)
        return tt, dt, blocks

    def loss(self, c: np.ndarray) -> tuple[float, np.ndarray]:
        """Normalized loss ``L / nu`` of the correlators ``c`` and its gradient with respect to ``c``."""
        t = np.tanh(self.alpha * c)                      # regularization acts on the correlators themselves
        q = float(np.mean(t * t))
        tt, dt, offset_blocks = self.relax(c)
        Jt = self.J @ tt
        value = (float(self.h @ tt) + 0.5 * float(tt @ Jt)) / self.nu + self.config.beta * q * q
        g = (self.h + Jt) / self.nu                     # dL/dt~
        if self.config.cardinality == "penalty":
            excess = np.array([tt[s].sum() for s in self.slices]) - self.targets
            value += 0.25 * self.rho * float(excess @ excess) / self.nu
            for s, e in zip(self.slices, excess):
                g[s] += 0.5 * self.rho * e / self.nu
        grad = g * dt
        for s in offset_blocks:
            # Implicit differentiation of the offset: db/dc_j = dt_j / sum_k dt_k, so the gradient of
            # the block is projected to keep its sum fixed.
            d = dt[s]
            grad[s] = d * (g[s] - float(g[s] @ d) / float(d.sum()))
        grad += self.config.beta * 2.0 * q * (2.0 * t / len(t)) * self.alpha * (1.0 - t * t)
        return value, grad

    def exact_loss(self, params: np.ndarray) -> float:
        """Loss of the exact correlators (one forward simulation, no gradient)."""
        return self.loss(self.circuit.correlators(params))[0]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

class _Tracker:
    """Counts the training cost, records the exact loss and reports progress."""

    def __init__(self, problem: PCEProblem, config: PCEConfig, progress: ProgressFn | None, expected: int):
        self.problem, self.config, self.progress = problem, config, progress
        self.expected = max(1, expected)
        self.loss_evaluations = self.circuit_executions = self.shots = 0
        self.history: list[dict] = []

    def count(self, executions: int = 1, evaluations: int = 1) -> None:
        """Account for ``executions`` circuit runs (each one measures every setting)."""
        self.loss_evaluations += evaluations
        self.circuit_executions += executions
        if self.config.training_shots:
            self.shots += executions * self.config.training_shots * len(self.problem.circuit.settings)

    def iteration(self, step: int, params: np.ndarray, estimate: float, exact: float | None = None) -> None:
        """Called once per optimizer iteration; logs the exact loss every ``log_every`` iterations."""
        every = self.config.log_every
        if every and (step == 1 or step % every == 0):
            value = exact if exact is not None else self.problem.exact_loss(params)
            self.history.append({"iteration": step, "evaluations": self.loss_evaluations,
                                 "loss": value, "loss_estimate": estimate})
        if self.progress is not None:
            self.progress(min(step / self.expected, 0.99))


def _train_adam(problem: PCEProblem, theta: np.ndarray, rng: np.random.Generator,
                tracker: _Tracker) -> tuple[np.ndarray, int, str]:
    """Adam (Kingma and Ba) with the default moments.

    Exact training uses the adjoint gradient, keeps the parameters with the
    lowest loss and stops on the patience criterion. Shot-based training
    estimates the loss and the parameter-shift gradient from samples, runs
    ``max_steps`` steps and returns the last parameters.
    """
    cfg, circuit = problem.config, problem.circuit
    b1, b2, eps = 0.9, 0.999, 1e-8
    m1, m2 = np.zeros_like(theta), np.zeros_like(theta)
    best_value, best_theta, losses = np.inf, theta.copy(), []
    for step in range(1, cfg.max_steps + 1):
        if cfg.training_shots is None:
            value, grad, _ = circuit.value_and_grad(theta, problem.loss)
            tracker.count()
            tracker.iteration(step, theta, value, exact=value)
            if value < best_value:
                best_value, best_theta = value, theta.copy()
            losses.append(value)
            if len(losses) > cfg.patience and losses[-1 - cfg.patience] - value < cfg.tolerance:
                return best_theta, step, "patience"
        else:
            c = circuit.estimate(theta, cfg.training_shots, rng)
            value, w = problem.loss(c)
            jac, executions = circuit.shifted_jacobian(theta, cfg.training_shots, rng)
            grad = jac.T @ w
            tracker.count(1 + executions)
            tracker.iteration(step, theta, value)
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * grad * grad
        theta = theta - cfg.learning_rate * (m1 / (1 - b1 ** step)) / (np.sqrt(m2 / (1 - b2 ** step)) + eps)
    if cfg.training_shots is None:
        return best_theta, cfg.max_steps, "max_steps"
    return theta, cfg.max_steps, "max_steps"


def _train_spsa(problem: PCEProblem, theta: np.ndarray, rng: np.random.Generator,
                tracker: _Tracker) -> tuple[np.ndarray, int, str]:
    """SPSA (Spall) with the standard gain sequences and a calibrated gain.

    ``a_k = a / (k + 1 + A)^alpha`` and ``c_k = c / (k + 1)^gamma`` with
    ``A = 0.1 * iterations``; ``a`` is set so that the first update moves the
    parameters by about ``spsa_target_step``. Every iteration perturbs all
    parameters at once along a random +-1 direction and uses two loss
    evaluations (exact or shot-based), whatever the number of parameters. The
    last parameters are returned.
    """
    cfg, circuit = problem.config, problem.circuit

    def loss(params: np.ndarray) -> float:
        tracker.count()
        return problem.loss(circuit.estimate(params, cfg.training_shots, rng))[0]

    n, iterations = len(theta), cfg.spsa_iterations
    stability = 0.1 * iterations
    slopes = []
    for _ in range(cfg.spsa_calibration):
        delta = rng.choice([-1.0, 1.0], n)
        slopes.append(abs(loss(theta + cfg.spsa_c * delta) - loss(theta - cfg.spsa_c * delta)) / (2 * cfg.spsa_c))
    gain = cfg.spsa_target_step * (stability + 1) ** cfg.spsa_alpha / max(float(np.mean(slopes)), 1e-12)
    for k in range(iterations):
        ak = gain / (k + 1 + stability) ** cfg.spsa_alpha
        ck = cfg.spsa_c / (k + 1) ** cfg.spsa_gamma
        delta = rng.choice([-1.0, 1.0], n)
        y_plus, y_minus = loss(theta + ck * delta), loss(theta - ck * delta)
        theta = theta - ak * (y_plus - y_minus) / (2 * ck) * delta
        tracker.iteration(k + 1, theta, 0.5 * (y_plus + y_minus))
    return theta, iterations, "iterations"


def _train_slsqp(problem: PCEProblem, theta: np.ndarray, rng: np.random.Generator,
                 tracker: _Tracker) -> tuple[np.ndarray, int, str]:
    """SciPy's SLSQP with exact or shot-based loss and gradient (parameter-shift rule)."""
    cfg, circuit = problem.config, problem.circuit
    step = {"k": 0, "last": np.nan}

    def fun(params: np.ndarray) -> tuple[float, np.ndarray]:
        if cfg.training_shots is None:
            value, grad, _ = circuit.value_and_grad(params, problem.loss)
            tracker.count()
        else:
            c = circuit.estimate(params, cfg.training_shots, rng)
            value, w = problem.loss(c)
            jac, executions = circuit.shifted_jacobian(params, cfg.training_shots, rng)
            grad = jac.T @ w
            tracker.count(1 + executions)
        step["last"] = value
        return value, grad

    def callback(params: np.ndarray) -> None:
        step["k"] += 1
        tracker.iteration(step["k"], params, step["last"])

    res = minimize(fun, theta, jac=True, method="SLSQP", callback=callback, options={"maxiter": cfg.max_steps})
    return np.asarray(res.x, dtype=float), int(res.nit), str(res.message)


def circuit_resources(m: int, config: PCEConfig) -> dict[str, int | float]:
    """Qubits, gates, depth and measurement settings of the PCE circuit for ``m`` variables.

    The depth counts layers of gates: each brickwork layer has one layer of
    single-qubit rotations and one layer of MS gates; each heavy-hex layer has
    two layers of rotations (``RY``, ``RZ``) and one layer of ``RZZ`` gates.
    """
    n, layers, _ = resolve(m, config)
    gates, n_params = build_ansatz(config.ansatz, n, layers)
    two = sum(g.kind in ("MS", "ZZ") for g in gates)
    settings = {a for a, _ in pauli_strings(m, config.k, n)}
    depth = (3 if config.ansatz == "heavy_hex" else 2) * layers
    return {"variables": m, "k": config.k, "qubits": n, "ansatz": config.ansatz, "layers": layers,
            "parameters": n_params, "single_qubit_gates": len(gates) - two, "two_qubit_gates": two,
            "depth": depth, "measurement_settings": len(settings), "compression": round(m / n, 2)}


def seed_streams(seed) -> tuple[np.random.SeedSequence, np.random.SeedSequence, np.random.SeedSequence]:
    """Three independent seed sequences (solver, baselines, reference) derived from ``seed`` without mutating it."""
    seq = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    words = seq.generate_state(12)
    return tuple(np.random.SeedSequence(words[4 * i:4 * i + 4]) for i in range(3))


def run_pce(
    qubo: QUBO,
    blocks: list[Block],
    config: PCEConfig,
    seed=0,
    reference: Reference | str = "exact",
    reference_restarts: int = 10,
    progress: ProgressFn | None = None,
    project: Callable[[np.ndarray], np.ndarray] | None = None,
    search: Callable[[np.ndarray], tuple[np.ndarray, float, int]] | None = None,
    greedy: Callable[[np.random.Generator], tuple[np.ndarray, float, int]] | None = None,
) -> PCEResult:
    """Minimize a block-constrained QUBO with the Pauli-correlation-encoding solver.

    The initial parameters are the first draw of the solver's random stream,
    uniform in ``[-pi, pi)``, so runs with the same ``seed`` and different
    optimizers start from the same point (paired comparison). The baselines
    and the reference use their own streams derived from ``seed``, so they are
    identical for every optimizer.

    Args:
        qubo: QUBO to minimize; its variables are the concatenated blocks.
        blocks: Block layout.
        config: Solver hyper-parameters.
        seed: Integer, sequence of integers or ``SeedSequence``.
        reference: A precomputed :class:`qopt.feasible.Reference`, or
            ``"exact"`` (enumeration) or ``"heuristic"`` (best of
            ``reference_restarts`` local searches).
        reference_restarts: Restarts of the heuristic reference.
        progress: Optional callback receiving the training progress in ``[0, 1]``.
        project: Optional map applied to every bitstring before it is scored
            and returned, e.g. setting auxiliary variables to their best value
            for the other variables. The reference must be built consistently.
        search: Optional post-processing ``bits -> (bits, energy, evaluations)``
            that keeps feasibility, used after decoding and in the random and
            untrained baselines (default: ``search_rounds`` rounds of
            :func:`qopt.feasible.swap_search`).
        greedy: Optional full local search ``rng -> (bits, energy, evaluations)``
            used as the classical baseline (default:
            :func:`qopt.feasible.greedy_search` from a random feasible bitstring).

    Returns:
        A :class:`PCEResult`.
    """
    t0 = time.time()
    if config.optimizer not in ("adam", "spsa", "slsqp", "exact"):
        raise ValueError(f"unknown optimizer {config.optimizer!r}; use 'adam', 'spsa', 'slsqp' or 'exact'")
    seq_solver, seq_base, seq_ref = seed_streams(seed)
    rng, base_rng = np.random.default_rng(seq_solver), np.random.default_rng(seq_base)
    if isinstance(reference, str):
        reference = (exact_reference(qubo, blocks) if reference == "exact"
                     else heuristic_reference(qubo, blocks, np.random.default_rng(seq_ref), reference_restarts))
    rounds = config.search_rounds

    def scored(bits: np.ndarray) -> tuple[np.ndarray, float]:
        bits = project(bits) if project is not None else bits
        return bits, float(qubo.energy(bits))

    if search is None:
        search = lambda bits: swap_search(qubo, bits, blocks, rounds)
    if greedy is None:
        greedy = lambda generator: greedy_search(qubo, blocks, generator)

    # Classical baselines, identical for every optimizer.
    random_energy = scored(search(random_feasible(blocks, base_rng))[0])[1]
    greedy_bits, _, greedy_evals = greedy(base_rng)
    greedy_energy = scored(greedy_bits)[1]

    if config.optimizer == "exact":
        if reference.kind != "exact":
            raise ValueError("optimizer='exact' needs an exact reference")
        bits = raw = reference.bits
        return PCEResult(bits, reference.energy, raw, reference.energy, False, random_energy, np.nan, np.nan,
                         greedy_energy, greedy_evals, reference, 0, 0, 0, np.nan, 0, 0, 0, 0, 0, 0, np.nan,
                         np.nan, "exact", time.time() - t0)

    problem = PCEProblem(qubo, blocks, config)
    circuit, shots = problem.circuit, config.readout_shots
    theta = rng.uniform(-np.pi, np.pi, circuit.n_params)

    # Untrained circuit, read and decoded like the trained one.
    initial_raw = decode(circuit.estimate(theta, shots, base_rng), blocks)
    initial_raw_energy = scored(initial_raw)[1]
    initial_energy = scored(search(initial_raw)[0])[1]

    budget = config.spsa_iterations if config.optimizer == "spsa" else config.max_steps
    tracker = _Tracker(problem, config, progress, config.expected_steps or budget)
    train = {"adam": _train_adam, "spsa": _train_spsa, "slsqp": _train_slsqp}[config.optimizer]
    theta, steps, stop_reason = train(problem, theta, rng, tracker)

    probs = circuit.probabilities(theta)
    exact_c = circuit.correlators_from_probabilities(probs)
    c = circuit.sample(probs, shots, rng)
    raw = decode(c, blocks)
    bits, _, search_evals = search(raw)
    bits, energy = scored(bits)
    raw, raw_energy = scored(raw)
    if progress is not None:
        progress(1.0)
    return PCEResult(
        bits=bits, energy=energy, raw_bits=raw, raw_energy=raw_energy,
        sign_feasible=is_feasible((c > 0).astype(np.int8), blocks),
        random_energy=random_energy, initial_raw_energy=initial_raw_energy, initial_energy=initial_energy,
        greedy_energy=greedy_energy, greedy_evaluations=greedy_evals, reference=reference,
        n_qubits=problem.n_qubits, layers=problem.layers, n_params=circuit.n_params,
        loss=problem.loss(exact_c)[0], steps=steps, loss_evaluations=tracker.loss_evaluations,
        circuit_executions=tracker.circuit_executions, training_shots_used=tracker.shots,
        readout_shots_used=(shots or 0) * len(circuit.settings), search_evaluations=search_evals,
        correlator_mean_abs=float(np.mean(np.abs(exact_c))), penalty=problem.rho, stop_reason=stop_reason,
        seconds=time.time() - t0, params=np.asarray(theta, dtype=float), history=tracker.history,
    )
