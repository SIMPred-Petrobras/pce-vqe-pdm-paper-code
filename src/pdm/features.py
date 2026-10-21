"""Sixteen vibration features per window: time domain and spectrum.

Notation for a window ``x[n]``, ``n = 1..N``: mean ``x_bar``, standard
deviation ``sigma`` (population), peak ``x_p = max |x[n]|``, power spectrum
``P[k] = |X[k]|^2`` at frequencies ``f_k`` and normalized power spectrum
``p_k = P[k] / sum P``.

Time domain (10)
    rms, peak, peak_to_peak, crest_factor (x_p / RMS), shape_factor
    (RMS / mean|x|), impulse_factor (x_p / mean|x|), clearance_factor
    (x_p / mean(sqrt|x|)^2), margin_factor (x_p / RMS^2), kurtosis, skewness.

Spectrum (6)
    band_energy (sum of ``P[k]`` in ``band_hz``), spectral_centroid,
    spectral_spread, spectral_skewness, spectral_kurtosis (moments of
    ``f`` under ``p``) and spectral_entropy (``-sum p log p``). ``X`` is the
    FFT of the mean-removed window multiplied by a Hann window. The six
    features share this FFT, which a device computes once if any of them is
    used.

``margin_factor`` is ``x_p / RMS^2``; part of the literature defines it as
``x_p / mean(sqrt|x|)^2``, which coincides with the clearance factor.

The cost of every feature on an ESP32-C6 microcontroller is modeled in
:mod:`pdm.costs`, which follows the same definitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .paderborn import SAMPLING_RATE

TIME_FEATURES = [
    "rms", "peak", "peak_to_peak", "crest_factor", "shape_factor", "impulse_factor",
    "clearance_factor", "margin_factor", "kurtosis", "skewness",
]
SPECTRUM_FEATURES = [
    "band_energy", "spectral_centroid", "spectral_spread",
    "spectral_skewness", "spectral_kurtosis", "spectral_entropy",
]

#: The 16 features, in table order (the order of the selection variables).
FEATURE_NAMES = TIME_FEATURES + SPECTRUM_FEATURES

#: Group of every feature. The ``spectrum`` features share the FFT; see :mod:`pdm.costs`.
FEATURE_GROUPS = {**{f: "time" for f in TIME_FEATURES}, **{f: "spectrum" for f in SPECTRUM_FEATURES}}

#: Version of the feature definitions; bump it whenever a formula changes, so
#: that cached feature tables are recomputed.
FEATURES_VERSION = 2


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters of the feature extraction.

    Attributes:
        window_seconds: Window length (s); 1 s = 64 000 samples.
        band_hz: Band of the band-energy feature (structural resonances).
    """

    window_seconds: float = 1.0
    band_hz: tuple[float, float] = (2000.0, 20000.0)

    def signature(self) -> dict:
        """Everything that changes the feature values (used to validate caches)."""
        return {"features_version": FEATURES_VERSION, "sampling_rate": SAMPLING_RATE,
                **{k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}}


def time_features(x: np.ndarray) -> dict[str, float]:
    """The ten time-domain features."""
    abs_x = np.abs(x)
    rms = np.sqrt(np.mean(x * x))
    peak = abs_x.max()
    mean_abs = abs_x.mean()
    z = (x - x.mean()) / x.std()
    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": x.max() - x.min(),
        "crest_factor": peak / rms,
        "shape_factor": rms / mean_abs,
        "impulse_factor": peak / mean_abs,
        "clearance_factor": peak / np.mean(np.sqrt(abs_x)) ** 2,
        "margin_factor": peak / rms ** 2,
        "kurtosis": np.mean(z ** 4),
        "skewness": np.mean(z ** 3),
    }


def spectrum_features(x: np.ndarray, fs: float, band_hz: tuple[float, float]) -> dict[str, float]:
    """The six spectrum features."""
    xw = (x - x.mean()) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(xw)) ** 2
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    p = power / power.sum()
    centroid = np.sum(f * p)
    spread = np.sqrt(np.sum((f - centroid) ** 2 * p))
    in_band = (f >= band_hz[0]) & (f <= band_hz[1])
    nonzero = p > 0
    return {
        "band_energy": power[in_band].sum(),
        "spectral_centroid": centroid,
        "spectral_spread": spread,
        "spectral_skewness": np.sum((f - centroid) ** 3 * p) / spread ** 3,
        "spectral_kurtosis": np.sum((f - centroid) ** 4 * p) / spread ** 4,
        "spectral_entropy": -np.sum(p[nonzero] * np.log(p[nonzero])),
    }


def window_features(x: np.ndarray, cfg: FeatureConfig, fs: float = SAMPLING_RATE) -> dict[str, float]:
    """All 16 features of one window, in :data:`FEATURE_NAMES` order.

    Args:
        x: Window samples (vibration).
        cfg: Feature parameters.
        fs: Sampling rate (Hz).
    """
    feats = {**time_features(x), **spectrum_features(x, fs, cfg.band_hz)}
    return {name: float(feats[name]) for name in FEATURE_NAMES}
