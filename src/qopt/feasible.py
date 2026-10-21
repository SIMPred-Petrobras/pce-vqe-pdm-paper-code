"""Feasible sets of block-constrained binary problems, references and local search.

The binary variables of a problem are split into consecutive *blocks*
(:class:`Block`). A block is either

* **constrained** -- every feasible bitstring has exactly ``weight`` ones in
  the block (a cardinality constraint), or
* **free** -- ``weight is None``, any assignment is allowed.

The one-class SVM of this study, for example, is a single constrained block
(exactly ``w = 32`` of 320 training windows); a problem with auxiliary
variables, such as feature selection with its ``y_fft`` variable, would add a
free block after the constrained one. (The VQE does not use blocks: its
circuit imposes the cardinality itself, see :mod:`qopt.vqe`.)

This module holds everything that does not depend on the quantum solver:

* :func:`exact_reference` -- exact enumeration of a small feasible set, which
  gives the optimum and the rank and approximation ratio of any answer;
* :func:`heuristic_reference` -- best of several classical local searches, the
  reference used when the feasible set is too large to enumerate;
* :func:`swap_search` -- rounds of first-improvement local search, the
  post-processing step of the PCE solver;
* :func:`greedy_search` -- steepest-descent local search until no move helps
  (a classical baseline).

Both local searches keep every bitstring feasible: inside a constrained block
they *swap* a one and a zero, inside a free block they *flip* single bits. Move
costs are evaluated in constant time from the local fields
``g = lin + quad @ x`` of the QUBO.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb, prod

import numpy as np

from .qubo import QUBO

#: Energies closer than this are considered equal (ties share the best rank).
TOL = 1e-9


@dataclass(frozen=True)
class Block:
    """A block of consecutive binary variables.

    Attributes:
        size: Number of variables in the block.
        weight: Number of ones in every feasible bitstring, or ``None`` for a
            free (unconstrained) block.
    """

    size: int
    weight: int | None = None

    def __post_init__(self) -> None:
        if self.weight is not None and not 0 <= self.weight <= self.size:
            raise ValueError(f"weight={self.weight} must be in [0, {self.size}]")

    @property
    def constrained(self) -> bool:
        """Whether the block has a cardinality constraint."""
        return self.weight is not None


def block_slices(blocks: list[Block]) -> list[slice]:
    """Positions of the variables of every block."""
    edges = np.cumsum([0] + [b.size for b in blocks])
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def is_feasible(bits: np.ndarray, blocks: list[Block]) -> bool:
    """Whether a bitstring satisfies the cardinality constraint of every block."""
    bits = np.asarray(bits)
    return all(int(bits[s].sum()) == b.weight for s, b in zip(block_slices(blocks), blocks) if b.constrained)


def feasible_count(blocks: list[Block]) -> int:
    """Number of feasible bitstrings (an exact Python integer)."""
    return prod(comb(b.size, b.weight) if b.constrained else 2 ** b.size for b in blocks)


def random_feasible(blocks: list[Block], rng: np.random.Generator) -> np.ndarray:
    """One uniformly random feasible bitstring."""
    parts = []
    for b in blocks:
        if b.constrained:
            part = np.zeros(b.size, dtype=np.int8)
            part[rng.choice(b.size, size=b.weight, replace=False)] = 1
        else:
            part = rng.integers(0, 2, size=b.size).astype(np.int8)
        parts.append(part)
    return np.concatenate(parts)


def _local_states(block: Block) -> np.ndarray:
    """All feasible assignments of one block, one per row."""
    if block.constrained:
        ones = np.array(list(combinations(range(block.size), block.weight)), dtype=np.int64)
        ones = ones.reshape(-1, block.weight)
        states = np.zeros((len(ones), block.size), dtype=np.int8)
        states[np.arange(len(ones))[:, None], ones] = 1
        return states
    codes = np.arange(2 ** block.size)
    return ((codes[:, None] >> np.arange(block.size - 1, -1, -1)) & 1).astype(np.int8)


def iter_feasible(blocks: list[Block], chunk: int = 1 << 17):
    """Yield every feasible bitstring, in chunks of at most ``chunk`` rows.

    The joint states are the Cartesian product of the local states of the
    blocks; they are generated chunk by chunk from mixed-radix indices, so
    memory stays bounded whatever the size of the feasible set.
    """
    local = [_local_states(b) for b in blocks]
    sizes = np.array([len(s) for s in local], dtype=np.int64)
    total = int(np.prod(sizes))
    for start in range(0, total, chunk):
        index = np.arange(start, min(start + chunk, total), dtype=np.int64)
        parts = []
        for states, size, stride in zip(local, sizes, np.cumprod(sizes[::-1])[::-1] // sizes):
            parts.append(states[(index // stride) % size])
        yield np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------

@dataclass
class Reference:
    """Reference energies used to score the answers of a solver.

    Every answer is scored by its *normalized gap*
    ``(E - E_ref) / (E_random - E_ref)``: 0 means as good as the reference,
    1 means as good as an average random feasible bitstring. When the feasible
    set was enumerated (``kind == "exact"``), the reference is the optimum and
    answers also get a rank (1 = optimum) and an approximation ratio
    ``(E_max - E) / (E_max - E_min)``.

    Attributes:
        kind: ``"exact"`` (enumeration) or ``"heuristic"`` (best of several
            classical local searches).
        energy: Optimum (exact) or best-known energy (heuristic).
        bits: Bitstring with that energy.
        random_mean: Mean energy of uniformly random feasible bitstrings.
        e_max: Largest feasible energy (exact only, ``nan`` otherwise).
        n_feasible: Number of feasible bitstrings.
        sorted_energies: All feasible energies in increasing order (exact only).
        evaluations: Energy evaluations spent to build the reference.
    """

    kind: str
    energy: float
    bits: np.ndarray
    random_mean: float
    e_max: float
    n_feasible: int
    sorted_energies: np.ndarray | None = None
    evaluations: int = 0

    def gap(self, energy: float) -> float:
        """Normalized gap ``(E - E_ref) / (E_random - E_ref)`` (0 = reference)."""
        span = self.random_mean - self.energy
        return 0.0 if span <= TOL else (energy - self.energy) / span

    def rank(self, energy: float) -> float:
        """Rank among all feasible bitstrings (1 = optimum; ``nan`` if not enumerated)."""
        if self.sorted_energies is None:
            return float("nan")
        return float(np.searchsorted(self.sorted_energies, energy - TOL, side="left") + 1)

    def ratio(self, energy: float) -> float:
        """Approximation ratio ``(E_max - E) / (E_max - E_min)`` (``nan`` if not enumerated)."""
        if self.sorted_energies is None:
            return float("nan")
        span = self.e_max - self.energy
        return 1.0 if span <= TOL else (self.e_max - energy) / span

    def is_best(self, energy: float) -> bool:
        """Whether an energy matches the reference (optimum or best known)."""
        return energy <= self.energy + TOL


def random_mean_energy(qubo: QUBO, blocks: list[Block], rng: np.random.Generator, samples: int = 2000) -> float:
    """Mean energy of ``samples`` uniformly random feasible bitstrings."""
    bits = np.stack([random_feasible(blocks, rng) for _ in range(samples)])
    return float(np.mean(qubo.energy(bits)))


def exact_reference(qubo: QUBO, blocks: list[Block], max_states: int = 5_000_000) -> Reference:
    """Optimum and all feasible energies, by enumeration.

    Args:
        qubo: QUBO to minimize.
        blocks: Block layout; the sizes must add up to ``qubo.n``.
        max_states: Refuse to enumerate larger feasible sets.

    Raises:
        ValueError: If the feasible set has more than ``max_states`` elements.
    """
    n = feasible_count(blocks)
    if n > max_states:
        raise ValueError(f"{n} feasible states exceed max_states={max_states}; use heuristic_reference")
    energies, best_bits, best = [], None, np.inf
    for bits in iter_feasible(blocks):
        e = qubo.energy(bits)
        j = int(np.argmin(e))
        if e[j] < best:
            best, best_bits = float(e[j]), bits[j].copy()
        energies.append(e)
    energies = np.sort(np.concatenate(energies))
    return Reference("exact", best, best_bits.astype(np.int8), float(energies.mean()), float(energies[-1]), n,
                     energies, n)


def heuristic_reference(qubo: QUBO, blocks: list[Block], rng: np.random.Generator, restarts: int = 10) -> Reference:
    """Best of ``restarts`` steepest-descent local searches from random starts.

    Used as the "best known" solution of problems too large to enumerate.
    """
    best_bits, best, evaluations = None, np.inf, 0
    for _ in range(restarts):
        bits, energy, evals = greedy_search(qubo, blocks, rng)
        evaluations += evals
        if energy < best:
            best, best_bits = energy, bits
    return Reference("heuristic", float(best), best_bits, random_mean_energy(qubo, blocks, rng), float("nan"),
                     feasible_count(blocks), None, evaluations)


# --------------------------------------------------------------------------
# Local search
# --------------------------------------------------------------------------

def _block_of(blocks: list[Block]) -> np.ndarray:
    """Block index of every variable."""
    return np.repeat(np.arange(len(blocks)), [b.size for b in blocks])


def swap_search(qubo: QUBO, bits: np.ndarray, blocks: list[Block], rounds: int = 1) -> tuple[np.ndarray, float, int]:
    """Rounds of first-improvement local search that keep the bitstring feasible.

    A round visits the variables in index order. A variable of a constrained
    block is swapped with the best variable of opposite value in the same
    block, and a variable of a free block is flipped, whenever the move lowers
    the energy. With one round this is the post-processing of the PCE paper
    (one sweep of single-bit flips) adapted to cardinality constraints.

    Args:
        qubo: QUBO to minimize.
        bits: Feasible starting bitstring.
        blocks: Block layout.
        rounds: Number of sweeps (stops early when a sweep changes nothing).

    Returns:
        ``(bits, energy, evaluations)``, where ``evaluations`` counts the
        candidate moves whose energy change was computed.
    """
    x = np.array(bits, dtype=np.int8)
    Q = np.asarray(qubo.quad, dtype=float)
    g = qubo.lin + Q @ x                                   # local fields
    block_of = _block_of(blocks)
    constrained = np.array([blocks[b].constrained for b in block_of])
    evaluations = 0
    for _ in range(rounds):
        changed = False
        for i in range(len(x)):
            d = 1 - 2 * int(x[i])                          # change of x_i (+1 or -1)
            if not constrained[i]:
                evaluations += 1
                if d * g[i] < -TOL:                        # single flip
                    x[i] += d
                    g += Q[:, i] * d
                    changed = True
                continue
            partners = np.flatnonzero((block_of == block_of[i]) & (x != x[i]))
            if not len(partners):
                continue
            delta = d * (g[i] - g[partners]) - Q[i, partners]   # x_i += d, x_j -= d
            evaluations += len(partners)
            j = int(np.argmin(delta))
            if delta[j] < -TOL:
                j = int(partners[j])
                x[i], x[j] = x[i] + d, x[j] - d
                g += (Q[:, i] - Q[:, j]) * d
                changed = True
        if not changed:
            break
    return x, float(qubo.energy(x)), evaluations


def greedy_search(qubo: QUBO, blocks: list[Block], rng: np.random.Generator,
                  bits: np.ndarray | None = None) -> tuple[np.ndarray, float, int]:
    """Steepest-descent local search inside the feasible set.

    Starting from ``bits`` (or a uniformly random feasible bitstring), every
    step evaluates all moves -- swaps of a one and a zero inside each
    constrained block and single flips inside each free block -- and applies
    the best one, until no move lowers the energy.

    Returns:
        ``(bits, energy, evaluations)``: the local optimum, its energy and the
        number of candidate moves evaluated.
    """
    x = random_feasible(blocks, rng) if bits is None else np.array(bits, dtype=np.int8)
    Q = np.asarray(qubo.quad, dtype=float)
    g = qubo.lin + Q @ x
    slices = block_slices(blocks)
    evaluations = 1
    while True:
        best_delta, best_move = -TOL, None
        for s, b in zip(slices, blocks):
            idx = np.arange(s.start, s.stop)
            if b.constrained:
                ones, zeros = idx[x[idx] == 1], idx[x[idx] == 0]
                if not len(ones) or not len(zeros):
                    continue
                # Moving one from i (in ones) to j (in zeros): -g_i + g_j - Q_ij.
                delta = g[zeros][None, :] - g[ones][:, None] - Q[np.ix_(ones, zeros)]
                evaluations += delta.size
                k = int(np.argmin(delta))
                if delta.flat[k] < best_delta:
                    best_delta = float(delta.flat[k])
                    best_move = (int(ones[k // len(zeros)]), int(zeros[k % len(zeros)]))
            else:
                delta = (1 - 2 * x[idx]) * g[idx]
                evaluations += len(idx)
                k = int(np.argmin(delta))
                if delta[k] < best_delta:
                    best_delta, best_move = float(delta[k]), (int(idx[k]),)
        if best_move is None:
            return x, float(qubo.energy(x)), evaluations
        if len(best_move) == 2:
            i, j = best_move
            x[i], x[j] = 0, 1
            g += Q[:, j] - Q[:, i]
        else:
            (i,) = best_move
            d = 1 - 2 * int(x[i])
            x[i] += d
            g += Q[:, i] * d
