"""mRMR feature selection under a cycle budget, written as a QUBO.

Variables
---------
``x_i = 1`` when feature ``i`` is selected (exactly ``M`` of the ``n``
features) and ``y = 1`` when the FFT shared by the spectrum features is
computed (one auxiliary variable, ``y_fft``).

Objective: minimum redundancy, maximum relevance (mRMR)
--------------------------------------------------------
::

    f(x) = - (1/M) sum_i a_i x_i  +  C(M, 2)^-1 sum_{i<j} B_ij x_i x_j

* ``a_i`` -- relevance: mutual information between feature ``i`` and the label
  "healthy training window" vs. "artificial-damage window", estimated by the
  k-nearest-neighbor method of scikit-learn and mapped to ``[0, 1]`` by the
  informational coefficient of correlation of Linfoot,
  ``a = sqrt(1 - exp(-2 I))`` (:func:`relevance`);
* ``B_ij`` -- redundancy: absolute Pearson correlation between features ``i``
  and ``j`` on the healthy training windows (:func:`redundancy`).

Both terms are means (over the selected features and over their pairs), so
they have the same scale whatever ``M``.

Constraints
-----------
* ``sum_i x_i = M`` -- imposed by the Hamming-weight-preserving ansatz of the
  VQE (:mod:`qopt.vqe`), not by the QUBO.
* ``y = OR_{i in spectrum} x_i`` -- the FFT is paid once if any spectrum
  feature is used: penalty ``P sum_{i in spectrum} x_i (1 - y)``, with ``P``
  just large enough that ``y = 0`` never pays off when a spectrum feature is
  used (a larger ``P`` would only raise the barrier between subsets with and
  without the FFT).
* ``cycles(x, y) = sum_i c_i x_i + C_fft y <= C_max`` -- the cycle budget on
  the device (:mod:`pdm.costs`). An inequality cannot be written as a
  quadratic penalty without extra (slack) variables. The **unbalanced
  penalization** of Montanez-Barrera et al. (2024) avoids them: with the
  normalized slack ``h = (C_max - cycles) / C_total``,

      zeta(h) = - lambda_1 h + lambda_2 h^2,

  which is quadratic in ``(x, y)``, grows quickly for ``h < 0`` (budget
  exceeded) and stays small for small ``h >= 0``. It is an approximation:
  ``zeta`` also grows for large positive slack, and its minimum is at
  ``h = lambda_1 / (2 lambda_2)``. When the budget binds, the best subsets
  just over it beat the constrained optimum on mRMR, and only a very steep
  penalty would push all of them above it, at the price of a landscape
  dominated by the penalty. ``(lambda_1, lambda_2)`` are therefore chosen on
  a grid (:func:`tune_penalty`) so that the constrained optimum is the
  lowest-energy *feasible* subset with as few infeasible subsets below it as
  possible, and the readout of the VQE keeps the best sample within the
  budget (:meth:`SelectionProblem.feasible`).

There are ``C(16, 5) = 4368`` subsets of 5 features, so the constrained
optimum is found by enumeration (:meth:`SelectionProblem.enumerate`), which
also checks the tuned penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import comb

import numpy as np
import pandas as pd

from qopt.qubo import QUBO

from .costs import SHARED_GROUP


def redundancy(X_healthy: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation between features on the healthy training windows (zero diagonal)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        B = np.abs(np.corrcoef(X_healthy, rowvar=False))
    B = np.nan_to_num(B)
    np.fill_diagonal(B, 0.0)
    return B


def relevance(X_healthy: np.ndarray, X_faulty: np.ndarray, seed: int = 0, n_neighbors: int = 3) -> np.ndarray:
    """Relevance of every feature: mutual information with the label healthy / faulty, in ``[0, 1]``.

    The mutual information ``I`` (in nats) is estimated by the
    k-nearest-neighbor method (Kraskov et al.; ``mutual_info_classif`` of
    scikit-learn) and mapped to ``[0, 1]`` by Linfoot's informational
    coefficient of correlation, ``sqrt(1 - exp(-2 I))``, which equals the
    absolute Pearson correlation for jointly Gaussian variables and is thus
    on the scale of the redundancy.
    """
    from sklearn.feature_selection import mutual_info_classif
    X = np.vstack([X_healthy, X_faulty])
    y = np.r_[np.zeros(len(X_healthy)), np.ones(len(X_faulty))]
    info = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=seed)
    return np.sqrt(1.0 - np.exp(-2.0 * np.maximum(info, 0.0)))


def all_subsets(n: int, k: int) -> np.ndarray:
    """Every subset of ``k`` of ``n`` items as 0/1 rows, shape ``(C(n, k), n)``, in lexicographic order."""
    rows = np.zeros((comb(n, k), n), dtype=np.int8)
    for r, idx in enumerate(combinations(range(n), k)):
        rows[r, list(idx)] = 1
    return rows


def subset_mask(bits) -> int:
    """Integer whose bit ``i`` is set when feature ``i`` is selected (a compact subset identifier)."""
    return int(sum(1 << i for i, b in enumerate(np.asarray(bits)) if b))


@dataclass
class SelectionProblem:
    """The selection QUBO of one training set.

    Attributes:
        names: Feature names (the order of the ``x`` variables).
        groups: Group of every feature.
        n_select: Number of features to select (``M``).
        a: Relevance of every feature.
        B: Redundancy matrix.
        own_cycles: Cycles per window of every feature (without the FFT).
        fft_cycles: Cycles per window of the FFT stage.
        total_cycles: Cycles of all features (the normalization of the slack).
        budget: Cycle budget ``C_max``.
        lambdas: ``(lambda_1, lambda_2)`` of the unbalanced penalization.
        link: Weight ``P`` of the ``y_fft`` link.
        qubo: The QUBO over ``(x_1..x_n, y)``.
    """

    names: list[str]
    groups: list[str]
    n_select: int
    a: np.ndarray
    B: np.ndarray
    own_cycles: np.ndarray
    fft_cycles: float
    total_cycles: float
    budget: float
    lambdas: tuple[float, float]
    link: float
    qubo: QUBO

    @property
    def n_features(self) -> int:
        return len(self.names)

    @property
    def spectral(self) -> np.ndarray:
        """Boolean mask of the features that need the FFT."""
        return np.array([g == SHARED_GROUP for g in self.groups])

    def canonical(self, x: np.ndarray) -> np.ndarray:
        """Bitstring(s) ``(x, y)`` with ``y`` set to its true value (1 if a spectrum feature is used)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.int8))
        y = (x[:, self.spectral].sum(axis=1) > 0).astype(np.int8)
        return np.hstack([x, y[:, None]])

    def parts(self, x: np.ndarray) -> pd.DataFrame:
        """Relevance, redundancy, mRMR objective, cycles, feasibility and QUBO energy of subsets ``x``.

        ``x`` holds rows of 0/1 over the features only; ``y`` takes its true
        value. ``objective`` is the mRMR term ``f`` alone; ``energy`` is the
        full QUBO (with the penalties) at the canonical ``(x, y)``.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        m = np.maximum(x.sum(axis=1), 1)
        pairs = np.maximum(m * (m - 1) / 2, 1)
        rel = x @ self.a / m
        red = 0.5 * np.einsum("si,ij,sj->s", x, self.B, x) / pairs
        uses_fft = x[:, self.spectral].sum(axis=1) > 0
        cycles = x @ self.own_cycles + uses_fft * self.fft_cycles
        return pd.DataFrame({"relevance": rel, "redundancy": red, "objective": red - rel, "cycles": cycles,
                             "feasible": cycles <= self.budget, "energy": self.qubo.energy(self.canonical(x))})

    def feasible(self, bits: np.ndarray) -> np.ndarray:
        """Whether bitstrings ``(x)`` or ``(x, y)`` (rows) fit the cycle budget, with the true ``y``."""
        x = np.atleast_2d(np.asarray(bits))[:, :self.n_features].astype(float)
        uses_fft = x[:, self.spectral].sum(axis=1) > 0
        return x @ self.own_cycles + uses_fft * self.fft_cycles <= self.budget

    def enumerate(self) -> tuple[np.ndarray, pd.DataFrame]:
        """Every subset of ``n_select`` features and its :meth:`parts`."""
        x = all_subsets(self.n_features, self.n_select)
        return x, self.parts(x)

    def optimum(self) -> tuple[int, int]:
        """Row indices (in :meth:`enumerate` order) of the constrained and of the unconstrained mRMR optima."""
        _, p = self.enumerate()
        obj = p["objective"].to_numpy()
        constrained = int(np.argmin(np.where(p["feasible"].to_numpy(), obj, np.inf)))
        return constrained, int(np.argmin(obj))

    def selected(self, x) -> list[str]:
        """Names of the selected features."""
        return [n for n, b in zip(self.names, np.asarray(x)[:self.n_features]) if b]


def _qubo(a, B, n_select, own, fft, budget, lambdas, link, spectral) -> QUBO:
    """QUBO over ``(x, y)``: mRMR + unbalanced penalization of the budget + ``y_fft`` link (normalized cycles)."""
    n = len(a)
    l1, l2 = lambdas
    lin = np.zeros(n + 1)
    quad = np.zeros((n + 1, n + 1))
    # mRMR: - (1/M) sum a_i x_i + C(M, 2)^-1 sum_{i<j} B_ij x_i x_j.
    lin[:n] = -np.asarray(a, dtype=float) / n_select
    Bq = np.array(B, dtype=float)
    np.fill_diagonal(Bq, 0.0)
    quad[:n, :n] = Bq / comb(n_select, 2) if n_select > 1 else 0.0
    # Unbalanced penalization of h = b - sum_k w_k z_k (z = (x, y)): -l1 h + l2 h^2, with z_k^2 = z_k.
    w = np.r_[own, fft]
    lin += l1 * w + l2 * (w * w - 2.0 * budget * w)
    quad += 2.0 * l2 * np.outer(w, w)
    np.fill_diagonal(quad, 0.0)
    const = -l1 * budget + l2 * budget * budget
    # Link: P x_i (1 - y) = P x_i - P x_i y for the spectrum features.
    lin[:n] += link * spectral
    quad[:n, n] -= link * spectral
    quad[n, :n] -= link * spectral
    return QUBO(lin, quad, const)


def build_problem(a: np.ndarray, B: np.ndarray, costs: pd.DataFrame, n_select: int, budget_cycles: float,
                  lambdas: tuple[float, float]) -> SelectionProblem:
    """The selection QUBO (see the module docstring) for given penalty weights.

    Args:
        a: Relevance of every feature.
        B: Redundancy matrix.
        costs: Cost table of :func:`pdm.costs.cost_table` (one row per feature, in the order of ``a`` and ``B``).
        n_select: Number of features to select.
        budget_cycles: Cycle budget per window.
        lambdas: ``(lambda_1, lambda_2)`` of the unbalanced penalization.
    """
    names, groups = list(costs["feature"]), list(costs["group"])
    total = float(costs.attrs["total_cycles"])
    own_cycles = costs["own_cycles"].to_numpy(dtype=float)
    fft_cycles = float(costs.attrs["fft_cycles"])
    own, fft, budget = own_cycles / total, fft_cycles / total, budget_cycles / total
    spectral = np.array([g == SHARED_GROUP for g in groups], dtype=float)
    # Largest change of the budget penalty when y flips (|h| <= 1 + budget); the link is twice that bound,
    # no larger: a heavy link would make dropping or adding the FFT a barrier for the VQE.
    l1, l2 = lambdas
    link = 2.0 * (l1 * fft + l2 * fft * (2.0 * (1.0 + budget) + fft))
    qubo = _qubo(a, B, n_select, own, fft, budget, lambdas, link, spectral)
    return SelectionProblem(names, groups, n_select, np.asarray(a, dtype=float), np.asarray(B, dtype=float),
                            own_cycles, fft_cycles, total, float(budget_cycles), (float(l1), float(l2)), float(link),
                            qubo)


#: Grid of the penalty weights searched by :func:`tune_penalty`.
LAMBDA1_GRID = tuple(float(v) for v in np.logspace(-3, 1, 17))
LAMBDA2_GRID = tuple(float(v) for v in np.logspace(-3, 2, 21))


def tune_penalty(a: np.ndarray, B: np.ndarray, costs: pd.DataFrame, n_select: int, budget_cycles: float,
                 grid1=LAMBDA1_GRID, grid2=LAMBDA2_GRID) -> tuple[SelectionProblem, dict]:
    """Choose ``(lambda_1, lambda_2)`` of the unbalanced penalization by enumeration.

    For every pair of the grid, the QUBO is evaluated on every subset (with
    its true ``y``). A pair is *valid* when the constrained mRMR optimum is
    the lowest-energy feasible subset. Among the valid pairs, the chosen one
    has the fewest infeasible subsets with a lower energy than the optimum
    (those are the traps that the feasible readout must reject) and, among
    those, the largest margin: the energy gap between the optimum and the
    second-best feasible subset, divided by the standard deviation of the
    energies. This tuning uses the enumeration, which is affordable here; for
    larger problems it would be done on smaller instances.

    Returns:
        The problem with the chosen weights and a summary (``lambda_1``,
        ``lambda_2``, ``valid_pairs``, ``infeasible_below``, ``margin``).
    """
    base = build_problem(a, B, costs, n_select, budget_cycles, (0.0, 0.0))
    x = all_subsets(base.n_features, n_select)
    feasible = base.feasible(x)
    target, _ = base.optimum()
    best, valid = None, 0
    for l1, l2 in product(grid1, grid2):
        problem = build_problem(a, B, costs, n_select, budget_cycles, (l1, l2))
        e = problem.qubo.energy(problem.canonical(x))
        ef = np.where(feasible, e, np.inf)
        if int(np.argmin(ef)) != target:
            continue
        valid += 1
        below = int(np.sum(~feasible & (e < e[target])))
        margin = float((np.partition(ef, 1)[1] - e[target]) / max(float(e.std()), 1e-12))
        key = (below, -margin)
        if best is None or key < best[0]:
            best = (key, problem)
    if best is None:
        raise ValueError("no penalty weights of the grid make the constrained optimum the best feasible subset")
    (below, neg_margin), problem = best
    return problem, {"lambda_1": problem.lambdas[0], "lambda_2": problem.lambdas[1], "valid_pairs": valid,
                     "infeasible_below": below, "margin": -neg_margin}
