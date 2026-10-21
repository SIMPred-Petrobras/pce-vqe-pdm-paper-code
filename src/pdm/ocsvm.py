"""One-class SVM with binary multipliers, trained as a QUBO on healthy data only.

Formulation
-----------
The one-class SVM of Scholkopf et al. (2001) solves the dual problem::

    min_alpha  1/2 sum_ij alpha_i alpha_j K_ij   s.t.  sum_i alpha_i = 1,  0 <= alpha_i <= 1/(nu n).

With binary multipliers ``alpha_i = lambda_i / w``, ``lambda_i in {0, 1}``
and exactly ``w`` ones:

* ``sum_i alpha_i = 1`` holds by construction;
* ``alpha_i <= 1/(nu n)`` holds for any ``nu <= w / n``;
* up to the factor ``1/w^2``, the objective is the QUBO
  ``1/2 sum_i K_ii lambda_i + sum_{i<j} K_ij lambda_i lambda_j``
  (:func:`ocsvm_qubo`), one block of ``n`` variables with ``w`` ones. Since
  ``K_ii = 1``, this chooses the ``w`` training windows with the smallest
  total pairwise similarity: ``w`` diverse prototypes of the healthy condition.

Kernel and decision
-------------------
The kernel needs no multiplier (Anguita et al., 2006)::

    K(u, v) = 2^(-gamma ||u - v||_1),

with ``gamma`` a power of two, chosen by the median rule on the healthy
training windows (:func:`auto_gamma`). The score of a window is the mean
kernel value over the support vectors, ``s(u) = (1/w) sum_{i in SV} K(u, x_i)``,
and a window is flagged as faulty when ``s(u) < tau``. In the notation of the
article, ``f(u) = sum_i lambda_i K(v_i, u) - rho = w (s(u) - tau)`` with
``rho = w tau``, so ``f(u) < 0`` and ``s(u) < tau`` are the same rule. The threshold ``tau``
is supervised (:func:`inner_threshold`): it maximizes the balanced accuracy
between healthy windows of bearings that the model did not see and windows
of artificial-damage bearings.

Integer-only inference
----------------------
:meth:`BinaryOCSVM.score_fixed` emulates an edge device without multiplier or
floating-point unit: features with ``in_bits`` fractional bits, the L1
distance in integers, the exponent ``gamma ||u - v||_1`` rounded to
``exp_bits`` fractional bits with shifts, ``2^-v`` as a lookup of
``2^-frac(v)`` in a table of ``2^exp_bits`` entries followed by a right shift
by ``int(v)``, and the sum over the support vectors as additions; ``w`` is a
power of two, so the mean is a shift, absorbed in the integer threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import OneClassSVM

from qopt.feasible import Block, greedy_search
from qopt.qubo import QUBO


def l1_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    """Gram matrix ``2^(-gamma ||a - b||_1)``, shape ``(len(A), len(B))``."""
    return np.exp2(-gamma * cdist(np.asarray(A, dtype=float), np.asarray(B, dtype=float), metric="cityblock"))


def auto_gamma(X: np.ndarray) -> float:
    """Power of two nearest to the inverse of the median pairwise L1 distance of ``X``.

    Typical kernel values are then close to ``2^-1``, and the kernel exponent
    is computed with shifts.
    """
    d = cdist(X, X, metric="cityblock")
    median = float(np.median(d[np.triu_indices_from(d, k=1)]))
    return float(2.0 ** np.round(np.log2(1.0 / max(median, 1e-12))))


def ocsvm_qubo(K: np.ndarray) -> QUBO:
    """QUBO ``1/2 sum_i K_ii l_i + sum_{i<j} K_ij l_i l_j`` of the one-class SVM dual (binary multipliers)."""
    K = np.asarray(K, dtype=float)
    quad = K.copy()
    np.fill_diagonal(quad, 0.0)
    return QUBO(0.5 * np.diag(K).copy(), quad, 0.0)


def solve_greedy(qubo: QUBO, w: int, rng: np.random.Generator, restarts: int = 1) -> tuple[np.ndarray, float]:
    """Best of ``restarts`` steepest-descent local searches for the one-class SVM QUBO (classical solver)."""
    best_bits, best = None, np.inf
    for _ in range(restarts):
        bits, energy, _ = greedy_search(qubo, [Block(qubo.n, w)], rng)
        if energy < best:
            best, best_bits = energy, bits
    return best_bits, best


def _fixed_kernel(Xq: np.ndarray, Sq: np.ndarray, log2_gamma: int, in_bits: int, out_bits: int,
                  exp_bits: int) -> np.ndarray:
    """Integer kernel ``round(2^out_bits 2^-v)``, ``v = gamma ||u - s||_1``, with shifts and one table lookup."""
    d = np.abs(Xq[:, None, :] - Sq[None, :, :]).sum(axis=-1)          # L1 distance, units of 2^-in_bits
    shift = log2_gamma - in_bits + exp_bits                              # v 2^exp_bits = d 2^shift
    v = d << shift if shift >= 0 else (d + (1 << (-shift - 1))) >> -shift
    table = np.rint(2.0 ** (out_bits - np.arange(2 ** exp_bits) / 2 ** exp_bits)).astype(np.int64)
    whole, frac = v >> exp_bits, v & (2 ** exp_bits - 1)
    return np.where(whole > out_bits, 0, np.right_shift(table[frac], np.minimum(whole, out_bits)))


@dataclass
class BinaryOCSVM:
    """One-class SVM defined by its ``w`` support vectors (binary multipliers).

    Attributes:
        support: Support vectors, shape ``(w, d)`` (scaled features).
        gamma: Kernel scale (a power of two).
    """

    support: np.ndarray
    gamma: float

    @classmethod
    def from_bits(cls, X: np.ndarray, bits: np.ndarray, gamma: float) -> "BinaryOCSVM":
        """Model whose support vectors are the training windows with ``bits == 1``."""
        return cls(np.asarray(X, dtype=float)[np.asarray(bits) == 1], gamma)

    @property
    def w(self) -> int:
        return len(self.support)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Mean kernel value over the support vectors (high = healthy)."""
        return l1_kernel(X, self.support, self.gamma).mean(axis=1)

    def score_fixed(self, X: np.ndarray, in_bits: int = 8, out_bits: int = 16, exp_bits: int = 6) -> np.ndarray:
        """Integer score: sum of the integer kernel values (units of ``2^-out_bits``).

        Divide by ``w 2^out_bits`` to compare with :meth:`score`; on the device
        the threshold is converted instead (:meth:`fixed_threshold`).
        """
        g = np.log2(self.gamma)
        if not float(g).is_integer():
            raise ValueError("integer inference needs gamma = 2^integer")
        quant = lambda A: np.rint(np.asarray(A, dtype=float) * 2 ** in_bits).astype(np.int64)
        return _fixed_kernel(quant(X), quant(self.support), int(g), in_bits, out_bits, exp_bits).sum(axis=1)

    def fixed_threshold(self, tau: float, out_bits: int = 16) -> int:
        """Threshold ``tau`` in the units of :meth:`score_fixed`."""
        return int(np.rint(tau * self.w * 2 ** out_bits))


def balanced_threshold(healthy_scores: np.ndarray, faulty_scores: np.ndarray) -> tuple[float, float]:
    """Threshold that maximizes the balanced accuracy of the rule "faulty if score < tau".

    The candidates are the midpoints between consecutive distinct scores; ties
    go to the lowest threshold.

    Returns:
        ``(tau, balanced accuracy on the given scores)``.
    """
    h = np.sort(np.asarray(healthy_scores, dtype=float))
    f = np.sort(np.asarray(faulty_scores, dtype=float))
    values = np.unique(np.concatenate([h, f]))
    candidates = np.concatenate([[values[0] - 1e-9], (values[:-1] + values[1:]) / 2, [values[-1] + 1e-9]])
    specificity = 1.0 - np.searchsorted(h, candidates, side="left") / len(h)    # healthy with score >= tau
    recall = np.searchsorted(f, candidates, side="left") / len(f)               # faulty with score < tau
    balanced = (specificity + recall) / 2
    best = int(np.argmax(balanced))
    return float(candidates[best]), float(balanced[best])


def inner_threshold(X: np.ndarray, bearing: np.ndarray, X_faulty: np.ndarray, w: int, gamma: float,
                    rng: np.random.Generator, restarts: int = 1) -> tuple[float, float]:
    """Supervised threshold of the binary one-class SVM (leave-one-bearing-out, artificial damage).

    For every training bearing ``b``, a model with ``w`` support vectors is
    trained by the classical local search (:func:`solve_greedy`) on the
    healthy windows of the other bearings; it scores the windows of ``b``,
    which it never saw, and the faulty calibration windows ``X_faulty``
    (artificial damage). The threshold maximizes the balanced accuracy over
    the pooled scores (:func:`balanced_threshold`). The test bearings are
    never used.

    Returns:
        ``(tau, validation balanced accuracy)``.
    """
    bearing = np.asarray(bearing)
    healthy, faulty = [], []
    for b in sorted(set(bearing)):
        train = bearing != b
        bits, _ = solve_greedy(ocsvm_qubo(l1_kernel(X[train], X[train], gamma)), w, rng, restarts)
        model = BinaryOCSVM.from_bits(X[train], bits, gamma)
        healthy.append(model.score(X[~train]))
        faulty.append(model.score(X_faulty))
    return balanced_threshold(np.concatenate(healthy), np.concatenate(faulty))


def sklearn_ocsvm(X: np.ndarray, gamma: float, nu: float) -> tuple[OneClassSVM, callable]:
    """scikit-learn ``OneClassSVM`` (continuous multipliers) with the same kernel.

    Its score is the multiplier-weighted mean kernel value over the training
    windows, ``sum_i alpha_i K(u, x_i) / sum_i alpha_i``: the analog of
    :meth:`BinaryOCSVM.score`. Unlike ``decision_function``, it has no
    model-specific offset and does not scale with ``nu n``, so a threshold
    estimated from other models (:func:`sklearn_inner_threshold`) applies to it.

    Returns:
        The fitted model and its scoring function ``X -> scores``.
    """
    model = OneClassSVM(kernel="precomputed", nu=nu).fit(l1_kernel(X, X, gamma))
    total = float(np.sum(model.dual_coef_))
    return model, (lambda U: model.score_samples(l1_kernel(U, X, gamma)) / total)


def sklearn_inner_threshold(X: np.ndarray, bearing: np.ndarray, X_faulty: np.ndarray, gamma: float,
                            nu: float) -> tuple[float, float]:
    """Supervised threshold of the scikit-learn reference, by the procedure of :func:`inner_threshold`."""
    bearing = np.asarray(bearing)
    healthy, faulty = [], []
    for b in sorted(set(bearing)):
        train = bearing != b
        _, score = sklearn_ocsvm(X[train], gamma, nu)
        healthy.append(score(X[~train]))
        faulty.append(score(X_faulty))
    return balanced_threshold(np.concatenate(healthy), np.concatenate(faulty))
