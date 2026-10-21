"""Cost of the features and of the detector on an ESP32-C6 microcontroller, in clock cycles.

Target device
-------------
The ESP32-C6 has a single-issue 32-bit RISC-V core (RV32IMAC, up to 160 MHz)
with a hardware multiplier and divider and **no floating-point unit**:
floating-point arithmetic is emulated in software and is roughly an order of
magnitude slower. Every feature is therefore modeled in fixed point:

* samples are 16-bit integers (Q15); products of two samples fit in 32 bits,
  and sums over a window are kept in 64-bit accumulators;
* the few divisions, square roots and logarithms per window (to finish a
  statistic) use integer routines; the per-sample square root of the
  clearance factor uses an integer square root of a 16-bit value, and the
  per-bin logarithm of the spectral entropy a fixed-point ``log2``
  (normalization by shifts and a small table);
* the FFT is a radix-2 fixed-point (Q15) FFT; the FFT of a real window of
  ``L`` points is computed as a complex FFT of ``L/2`` points plus a split
  step.

Cycle model
-----------
The cost of a feature is the number of clock cycles per window, counted
analytically from its definition in :mod:`pdm.features` as loops over the
samples (or over the spectrum bins), each iteration costing the cycles of its
operations (:data:`CYCLES`) plus the loop overhead. The cycle counts are
estimates for a single-issue in-order RISC-V core, not measurements:

=====================  =======  ================================================
operation              cycles   meaning
=====================  =======  ================================================
``load``               2        16/32-bit load (with the load-use stall)
``store``              1        store
``alu``                1        add, subtract, shift, compare, logic
``mul``                2        32-bit multiply (low or high word)
``div``                33       32-bit integer division (iterative divider)
``div64``              150      64-bit division (software routine)
``isqrt16``            32       integer square root of a 16-bit value
``isqrt32``            64       integer square root of a 32-bit value
``log2q``              30       fixed-point log2 (normalize, table, interpolate)
``loop``               2        loop overhead per iteration (increment, branch)
``butterfly``          44       radix-2 Q15 FFT butterfly, all included
=====================  =======  ================================================

The butterfly cost is calibrated on the ESP-DSP library of Espressif: its
portable C (ANSI) implementation of the Q15 radix-2 FFT,
``dsps_fft2r_sc16``, takes about 225 000 cycles for 1024 points on the
Espressif cores without DSP extensions, i.e. about 44 cycles for each of the
``(L/2) log2 L`` butterflies. Composite costs: a 64-bit accumulation is 3
``alu`` (add, carry, add), a 64-bit product 2 ``mul``, an absolute value 3
``alu`` (branch-free).

Shared stage
------------
The six spectrum features share one stage, paid once if any of them is
used: mean removal, Hann window, zero padding to ``L = 2^ceil(log2 N)``
points (65 536 for ``N = 64 000``), the real FFT, the power spectrum and its
sum. So the cost of a subset ``S`` is::

    cycles(S) = sum_{i in S} c_i + C_fft * [S uses a spectrum feature].

Time-domain features have no shared stage: each counts its own passes over
the window (a device computing several of them would merge some loops; the
model is conservative).
"""

from __future__ import annotations

from collections import Counter
from math import ceil, floor, log2

import pandas as pd

from .features import FEATURE_GROUPS, FEATURE_NAMES, FeatureConfig
from .paderborn import SAMPLING_RATE

#: Cycles of every elementary operation (see the module docstring).
CYCLES = {"load": 2, "store": 1, "alu": 1, "mul": 2, "div": 33, "div64": 150, "isqrt16": 32, "isqrt32": 64,
          "log2q": 30, "loop": 2, "butterfly": 44}

#: Clock frequency of the ESP32-C6 (Hz), used to convert cycles to time.
CLOCK_HZ = 160e6

#: The group whose features share a stage (the FFT).
SHARED_GROUP = "spectrum"


def _sizes(cfg: FeatureConfig, fs: float = SAMPLING_RATE) -> dict[str, int]:
    """Window length ``N``, FFT length ``L``, spectrum bins ``K`` and bins inside the energy band."""
    n = int(round(cfg.window_seconds * fs))
    L = 1 << ceil(log2(n))
    resolution = fs / L
    return {"N": n, "L": L, "K": L // 2 + 1,
            "K_band": int(floor((cfg.band_hz[1] - cfg.band_hz[0]) / resolution)) + 1}


def _loop(n: int, **ops: float) -> Counter:
    """Operations of a loop of ``n`` iterations, each with the given operations plus the loop overhead."""
    return Counter({op: n * count for op, count in {**ops, "loop": 1}.items()})


def operation_counts(cfg: FeatureConfig = FeatureConfig(),
                     fs: float = SAMPLING_RATE) -> tuple[dict[str, Counter], Counter]:
    """Operation counts per window of every feature (own cost) and of the shared FFT stage.

    Returns:
        ``(features, fft_stage)``: :class:`collections.Counter` objects mapping
        operation names (keys of :data:`CYCLES`) to counts.
    """
    s = _sizes(cfg, fs)
    N, L, K = s["N"], s["L"], s["K"]
    acc64, absv, maxv = 3, 3, 2                     # alu operations of a 64-bit accumulation, |x| and a running max

    sum_sq = dict(load=1, mul=1, alu=acc64)                      # sum x^2
    peak = dict(load=1, alu=absv + maxv)                         # max |x|
    feats = {
        "rms": _loop(N, **sum_sq) + Counter(div64=1, isqrt32=1),
        "peak": _loop(N, **peak),
        "peak_to_peak": _loop(N, load=1, alu=2 * maxv) + Counter(alu=1),
        "crest_factor": _loop(N, load=1, mul=1, alu=acc64 + absv + maxv) + Counter(div64=1, isqrt32=1, div=1),
        "shape_factor": _loop(N, load=1, mul=1, alu=acc64 + absv + acc64) + Counter(div64=2, isqrt32=1, div=1),
        "impulse_factor": _loop(N, load=1, alu=absv + maxv + acc64) + Counter(div64=1, div=1),
        "clearance_factor": _loop(N, load=1, isqrt16=1, alu=absv + maxv + acc64) + Counter(div64=1, mul=1, div=1),
        "margin_factor": _loop(N, load=1, mul=1, alu=acc64 + absv + maxv) + Counter(div64=1, div=1),
        # Two passes: the mean, then the centered powers d^2 and d^4 (or d^3) in 64 bits.
        "kurtosis": _loop(N, load=1, alu=acc64) + _loop(N, load=1, mul=1 + 2, alu=1 + 2 * acc64 + 1)
                    + Counter(div64=4),
        "skewness": _loop(N, load=1, alu=acc64) + _loop(N, load=1, mul=1 + 2, alu=1 + 2 * acc64 + 1)
                    + Counter(div64=4, isqrt32=1),
    }
    # Spectrum features, given the power spectrum P[k] and its sum (bin index k stands for the frequency).
    centroid = _loop(K, load=1, mul=2, alu=4) + Counter(div64=1)             # sum k P[k] in 64 bits
    second = dict(load=1, mul=1 + 2, alu=1 + 4)                              # (k - mu)^2 P[k]
    feats.update({
        "band_energy": _loop(s["K_band"], load=1, alu=acc64),
        "spectral_centroid": centroid,
        "spectral_spread": centroid + _loop(K, **second) + Counter(div64=1, isqrt32=1),
        "spectral_skewness": centroid + _loop(K, load=1, mul=1 + 2 + 2 + 2, alu=1 + 4 + 4)
                             + Counter(div64=3, isqrt32=1),
        "spectral_kurtosis": centroid + _loop(K, load=1, mul=1 + 2 + 2, alu=1 + 4 + 4) + Counter(div64=3),
        "spectral_entropy": _loop(K, load=1, log2q=1, mul=2, alu=4) + Counter(log2q=1, div64=1),
    })
    butterflies = (L // 4) * int(log2(L // 2))                               # complex FFT of L/2 points
    fft_stage = (_loop(N, load=1, alu=acc64) + Counter(div64=1)              # mean
                 + _loop(N, load=2, mul=1, alu=2, store=1)                   # remove the mean, Hann window
                 + _loop(L - N, store=1)                                     # zero padding
                 + Counter(butterfly=butterflies)
                 + _loop(L // 2, load=4, mul=4, alu=8, store=4)              # split step of the real FFT
                 + _loop(K, load=2, mul=2, alu=1, store=1)                   # power spectrum |X|^2
                 + _loop(K, load=1, alu=acc64))                              # its sum
    return feats, fft_stage


def cycles(ops: Counter, table: dict[str, float] = CYCLES) -> float:
    """Clock cycles of an operation count."""
    return float(sum(table[op] * count for op, count in ops.items()))


def cost_table(cfg: FeatureConfig = FeatureConfig(), table: dict[str, float] = CYCLES) -> pd.DataFrame:
    """Cycles per window of every feature.

    Columns: ``feature``, ``group``, ``own_cycles`` (the feature itself),
    ``stage_cycles`` (the FFT stage, for spectrum features), ``standalone_cycles``
    (the feature computed alone), the same normalized by the cycles of all 16
    features (``own``, ``stage``, ``standalone``) and the operation counts
    (``n_<op>``). ``attrs`` holds ``total_cycles`` and ``fft_cycles``.
    """
    feats, fft_stage = operation_counts(cfg)
    fft = cycles(fft_stage, table)
    total = sum(cycles(feats[f], table) for f in FEATURE_NAMES) + fft
    rows = []
    for f in FEATURE_NAMES:
        own = cycles(feats[f], table)
        stage = fft if FEATURE_GROUPS[f] == SHARED_GROUP else 0.0
        rows.append({"feature": f, "group": FEATURE_GROUPS[f], "own_cycles": own, "stage_cycles": stage,
                     "standalone_cycles": own + stage, "own": own / total, "stage": stage / total,
                     "standalone": (own + stage) / total, **{f"n_{op}": feats[f].get(op, 0) for op in CYCLES}})
    out = pd.DataFrame(rows)
    out.attrs.update(total_cycles=total, fft_cycles=fft)
    return out


def fft_table(cfg: FeatureConfig = FeatureConfig(), table: dict[str, float] = CYCLES) -> pd.DataFrame:
    """Cycles of the shared FFT stage, with its operation counts."""
    _, fft_stage = operation_counts(cfg)
    total = cost_table(cfg, table).attrs["total_cycles"]
    return pd.DataFrame([{"stage": "fft", "cycles": cycles(fft_stage, table), "cost": cycles(fft_stage, table) / total,
                          **{f"n_{op}": fft_stage.get(op, 0) for op in CYCLES}}])


def ocsvm_inference_cycles(n_support: int, n_features: int, table: dict[str, float] = CYCLES) -> float:
    """Cycles of the integer-only one-class SVM decision for one window.

    Per support vector: the L1 distance (per feature: load, subtract,
    absolute value, add), the kernel exponent and the power of two (two
    shifts and one table lookup) and one accumulation; then one comparison
    with the threshold. The mean over the support vectors is a shift (``w`` is
    a power of two) absorbed in the threshold.
    """
    per_feature = Counter(load=1, alu=1 + 3 + 1)
    per_sv = Counter({k: v * n_features for k, v in per_feature.items()}) + Counter(alu=3, load=1, loop=1)
    return cycles(Counter({k: v * n_support for k, v in per_sv.items()}) + Counter(alu=1), table)
