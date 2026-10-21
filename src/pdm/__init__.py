"""Predictive-maintenance application: bearing data, features, costs, feature selection and one-class SVM.

Modules:

* :mod:`pdm.download` -- download and extraction of the Paderborn bearing dataset.
* :mod:`pdm.paderborn` -- bearing table (damage, manufacturer, geometry),
  reading of the measurements and non-overlapping windows.
* :mod:`pdm.protocol` -- bearing-wise cross-validation: rounds and test sets.
* :mod:`pdm.features` -- the 16 vibration features of a window (time domain and spectrum).
* :mod:`pdm.costs` -- clock-cycle model of the feature extraction and of the
  detector on an ESP32-C6 microcontroller (fixed point).
* :mod:`pdm.selection` -- mRMR feature selection under a cycle budget, as a QUBO.
* :mod:`pdm.ocsvm` -- one-class SVM with binary multipliers as a QUBO,
  multiplier-free kernel, threshold and integer-only inference.
* :mod:`pdm.metrics` -- detection metrics.
"""
