"""Bearing-wise cross-validation protocol: rounds, calibration faults and test sets.

Windows of the same bearing are never split between training and test: a
model could recognize the bearing instead of its condition. The protocol
therefore works with bearings:

* **Healthy folds.** The 6 healthy bearings (all IBU) form 3 folds of 2,
  balanced by run-in time (longest with shortest): {K001, K003},
  {K002, K004}, {K005, K006}. Round ``r`` tests on fold ``r`` and trains the
  one-class SVM on the healthy windows of the other 4 bearings.
* **What uses faults.** The one-class SVM is trained on healthy training
  windows only. Two steps use the windows of the 12 artificial-damage
  bearings (:func:`calibration_rows`), which are never used for testing: the
  relevance of the features (mRMR, :func:`pdm.selection.relevance`) and the
  decision threshold (:func:`pdm.ocsvm.inner_threshold`). Doing both without
  any faulty bearing is left as an open problem of the study.
* **Test set** (:func:`test_sets`): the healthy windows of the 2 test
  bearings against all windows of the 14 real-damage bearings, which are
  never used before the test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .paderborn import BEARINGS, codes

#: Healthy folds, balanced by run-in time (Table 7 of Lessmeier et al., 2016).
HEALTHY_FOLDS = (("K001", "K003"), ("K002", "K004"), ("K005", "K006"))
#: Artificial-damage bearings: the faulty windows of the feature relevance and of the threshold calibration.
ARTIFICIAL_BEARINGS = tuple(codes("artificial"))
#: Real-damage bearings (test set).
REAL_BEARINGS = tuple(codes("real"))


@dataclass(frozen=True)
class Round:
    """One cross-validation round.

    Attributes:
        index: Round number (0-based).
        train_bearings: Healthy bearings whose windows train the detector.
        test_bearings: Healthy bearings of the test set.
    """

    index: int
    train_bearings: tuple[str, ...]
    test_bearings: tuple[str, ...]


def make_rounds(folds=HEALTHY_FOLDS) -> list[Round]:
    """One round per fold: test on the fold, train on the others."""
    return [Round(r, tuple(b for f in folds if f is not fold for b in f), tuple(fold))
            for r, fold in enumerate(folds)]


def annotate(windows: pd.DataFrame) -> pd.DataFrame:
    """Add the bearing metadata (group, damage, manufacturer) to a table with a ``bearing`` column."""
    out = windows.copy()
    for col in ("group", "location", "method", "extent", "manufacturer"):
        out[col] = [getattr(BEARINGS[b], col) for b in out["bearing"]]
    return out


def test_sets(windows: pd.DataFrame, rnd: Round) -> dict[str, np.ndarray]:
    """Row labels of the test set of one round: ``{"healthy", "real"}``."""
    b = windows["bearing"]
    return {
        "healthy": windows.index[b.isin(rnd.test_bearings)].to_numpy(),
        "real": windows.index[b.isin(REAL_BEARINGS)].to_numpy(),
    }


def calibration_rows(windows: pd.DataFrame) -> np.ndarray:
    """Row labels of the artificial-damage windows (feature relevance and threshold calibration)."""
    return windows.index[windows["bearing"].isin(ARTIFICIAL_BEARINGS)].to_numpy()


def training_rows(windows: pd.DataFrame, rnd: Round, max_per_bearing: int | None = None,
                  seed: int = 0) -> np.ndarray:
    """Row labels of the healthy training windows of a round.

    Args:
        windows: Window table.
        rnd: The round.
        max_per_bearing: Keep at most this many windows per bearing (a random
            subset; used only by the short smoke-test profile).
        seed: Seed of that subset.
    """
    rows = []
    for b in rnd.train_bearings:
        idx = windows.index[windows["bearing"] == b].to_numpy()
        if max_per_bearing is not None and len(idx) > max_per_bearing:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 102, rnd.index]))
            idx = np.sort(rng.choice(idx, max_per_bearing, replace=False))
        rows.extend(idx)
    return np.array(rows)
