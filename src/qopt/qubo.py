"""Container for a quadratic unconstrained binary optimization (QUBO) problem.

The objective over binary variables ``x in {0, 1}^n`` is written as::

    f(x) = const + sum_i lin[i] x_i + sum_{i<j} quad[i, j] x_i x_j,

with ``quad`` symmetric and zero on the diagonal (linear terms, including the
diagonal of the usual upper-triangular QUBO matrix, live in ``lin`` because
``x_i**2 = x_i``). In the notation of the article, ``x^T Q x`` with ``Q``
upper triangular, ``lin[i] = Q_ii`` and ``quad[i, j] = quad[j, i] = Q_ij``.

Applications build a :class:`QUBO` (:func:`pdm.selection.build_problem` for
feature selection, :func:`pdm.ocsvm.ocsvm_qubo` for the one-class SVM) and
minimize it with :func:`qopt.vqe.run_vqe` or :func:`qopt.pce.run_pce`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QUBO:
    """QUBO objective ``const + lin . x + sum_{i<j} quad[i, j] x_i x_j``.

    Attributes:
        lin: Linear coefficients, shape ``(n,)``.
        quad: Symmetric pairwise coefficients with zero diagonal, shape ``(n, n)``.
        const: Constant offset (does not change the minimizer).
    """

    lin: np.ndarray
    quad: np.ndarray
    const: float = 0.0

    @property
    def n(self) -> int:
        """Number of binary variables."""
        return len(self.lin)

    def energy(self, bits: np.ndarray) -> np.ndarray:
        """Objective value of one bitstring ``(n,)`` or of a batch ``(m, n)``.

        The factor 1/2 compensates for counting each pair twice in the
        symmetric matrix product.
        """
        bits = np.asarray(bits, dtype=float)
        return self.const + bits @ self.lin + 0.5 * np.einsum("...i,ij,...j->...", bits, self.quad, bits)

    def ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Spin form of the objective, with ``x_i = (1 + s_i) / 2`` and ``s_i in {-1, +1}``.

        The article writes the Ising Hamiltonian with ``x_i = (1 - z_i) / 2``,
        where ``z_i`` is the eigenvalue of ``Z_i`` (``|0> -> +1``); the two
        conventions differ only by the sign of the spins, ``s_i = -z_i``,
        which flips the sign of ``h`` and leaves ``J`` unchanged.

        Returns:
            ``(h, J, offset)`` such that ``f(x) = offset + h . s + sum_{i<j}
            J[i, j] s_i s_j``; ``J`` is symmetric with zero diagonal.
        """
        quad = np.asarray(self.quad, dtype=float)
        h = 0.5 * self.lin + 0.25 * quad.sum(axis=1)
        offset = self.const + 0.5 * float(self.lin.sum()) + 0.125 * float(quad.sum())
        return h, 0.25 * quad, offset
