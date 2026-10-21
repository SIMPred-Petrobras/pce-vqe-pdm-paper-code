"""Units of work executed by the worker processes.

Every task is a top-level function ``task(spec, progress) -> result``:

* ``spec`` is a plain dictionary with everything the task needs (arrays,
  settings, seeds), built by :mod:`experiment.pipeline` in the main process;
* ``progress`` is a callback that receives the fraction of the task done;
* the result is a dictionary (or list) of rows that the main process writes to disk.

Tasks never write files and only the feature task reads the dataset, so they
can run in any process and in any order, and their results depend only on
their spec (every random choice is seeded from it).
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np

from pdm.features import FEATURE_NAMES, window_features
from pdm.metrics import detection_metrics
from pdm.ocsvm import (BinaryOCSVM, auto_gamma, inner_threshold, l1_kernel, ocsvm_qubo, sklearn_inner_threshold,
                       sklearn_ocsvm)
from pdm.paderborn import MeasurementReader, split_windows
from pdm.selection import subset_mask
from qopt.feasible import Block, heuristic_reference, random_feasible, swap_search
from qopt.pce import run_pce
from qopt.vqe import run_vqe

log = logging.getLogger("experiment")


def _rng(*words) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(w) for w in words]))


def evaluate(score, tau: float, tests: dict) -> dict:
    """Detection metrics on the test set (healthy test windows vs. real damage).

    Args:
        score: ``X -> scores`` (high = healthy).
        tau: Threshold (a window is flagged as faulty when its score is below it).
        tests: ``{"healthy": X, "real": X}`` (scaled, selected features).
    """
    healthy = score(tests["healthy"])
    real = score(tests["real"])
    return detection_metrics(np.r_[np.zeros(len(healthy)), np.ones(len(real))], np.r_[healthy, real], tau)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def features_task(spec: dict, progress) -> list[dict]:
    """Windows and features of all measurements of one bearing in one condition.

    A measurement that cannot be read is skipped with a warning; a measurement
    slightly shorter than 4 s gives only 3 windows, since windows never overlap.

    Spec keys: ``archives_dir``, ``raw_dir``, ``bearing``, ``condition``,
    ``measurements``, ``features`` (:class:`pdm.features.FeatureConfig`).
    """
    reader = MeasurementReader(spec["archives_dir"], spec["raw_dir"])
    cfg, code = spec["features"], spec["bearing"]
    rows = []
    for i, m in enumerate(spec["measurements"]):
        try:
            vibration, speed = reader.read(code, spec["condition"], m)
        except Exception as error:                     # corrupt file: skip it, keep the others
            log.warning(f"{code} {spec['condition']} measurement {m} could not be read ({error!r}); skipped")
            continue
        for j, x, rpm in split_windows(vibration, speed, cfg.window_seconds):
            rows.append({"condition": spec["condition"], "bearing": code, "measurement": m, "window": j,
                         "shaft_hz": rpm / 60.0, **window_features(x, cfg)})
        progress((i + 1) / len(spec["measurements"]))
    return rows


# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------

def subsets_task(spec: dict, progress) -> list[dict]:
    """Detection quality of scikit-learn's one-class SVM for a list of feature subsets.

    For each subset (a bit mask over :data:`pdm.features.FEATURE_NAMES`), a
    conventional one-class SVM (continuous multipliers, kernel
    ``2^(-gamma ||u - v||_1)``, ``nu``) is trained on the healthy training
    windows; its threshold maximizes the balanced accuracy between unseen
    healthy windows (leave-one-bearing-out) and the artificial-damage windows
    (:func:`pdm.ocsvm.sklearn_inner_threshold`). It is then evaluated on the
    test set of the round.

    Spec keys: ``round``, ``X_train`` (all 16 features), ``bearing``,
    ``X_faulty`` (artificial damage, all 16 features), ``tests`` (all 16
    features), ``masks``, ``nu``.
    """
    X, bearing, rows = spec["X_train"], spec["bearing"], []
    for i, mask in enumerate(spec["masks"]):
        cols = [c for c in range(len(FEATURE_NAMES)) if mask >> c & 1]
        Xs = X[:, cols]
        gamma = auto_gamma(Xs)
        tau, validation = sklearn_inner_threshold(Xs, bearing, spec["X_faulty"][:, cols], gamma, spec["nu"])
        _, score = sklearn_ocsvm(Xs, gamma, spec["nu"])
        tests = {k: v[:, cols] for k, v in spec["tests"].items()}
        metrics = evaluate(score, tau, tests)
        rows.append({"round": spec["round"], "mask": int(mask),
                     "features": json.dumps([FEATURE_NAMES[c] for c in cols]), "gamma": gamma, "tau": tau,
                     "validation_balanced_accuracy": validation, **metrics})
        if i % 20 == 0:
            progress((i + 1) / len(spec["masks"]))
    return rows


def vqe_task(spec: dict, progress) -> dict:
    """One VQE run on the feature-selection QUBO.

    Spec keys: ``keys`` (identifying columns), ``problem``
    (:class:`pdm.selection.SelectionProblem`), ``vqe`` (:class:`qopt.vqe.VQEConfig`),
    ``seed``, ``optimum_mask`` (the constrained optimum), ``optimum`` and
    ``random_mean`` (its mRMR objective and the mean objective of all
    subsets, for the normalized gap). The readout keeps the best sample within
    the cycle budget.
    """
    problem = spec["problem"]
    res = run_vqe(problem.qubo, problem.n_select, spec["vqe"], seed=spec["seed"], progress=progress,
                  accept=problem.feasible)
    x = res.bits[:problem.n_features]
    parts = problem.parts(x).iloc[0]
    energy = float(parts["energy"])
    span = spec["random_mean"] - spec["optimum"]
    row = {**spec["keys"], "mask": subset_mask(x), "features": json.dumps(problem.selected(x)),
           "sampled_energy": res.energy, "energy": energy, "objective": float(parts["objective"]),
           "relevance": float(parts["relevance"]), "redundancy": float(parts["redundancy"]),
           "cycles": float(parts["cycles"]), "feasible": bool(parts["feasible"]),
           "gap": (float(parts["objective"]) - spec["optimum"]) / span if span > 0 else 0.0,
           "is_optimum": subset_mask(x) == spec["optimum_mask"], "p_answer": res.p_answer,
           "most_likely_mask": subset_mask(res.most_likely[:problem.n_features]), "loss": res.loss,
           "steps": res.steps, "evaluations": res.evaluations, "training_shots": res.shots_used,
           "seconds": res.seconds}
    return {"row": row, "history": [{**spec["keys"], **h} for h in res.history]}


# --------------------------------------------------------------------------
# One-class SVM with binary multipliers
# --------------------------------------------------------------------------

def ocsvm_prep_task(spec: dict, progress) -> dict:
    """Threshold and best-known solution of the one-class SVM QUBO of one round.

    The threshold comes from leave-one-bearing-out models trained by the
    classical local search, maximizing the balanced accuracy between unseen
    healthy windows and the artificial-damage windows
    (:func:`pdm.ocsvm.inner_threshold`); the reference is the best of
    ``restarts`` classical local searches.

    Spec keys: ``X_train`` (selected features), ``bearing``, ``X_faulty``
    (artificial damage, selected features), ``w``, ``gamma``,
    ``inner_restarts``, ``restarts``, ``seed``.
    """
    X, w, gamma = spec["X_train"], spec["w"], spec["gamma"]
    tau, validation = inner_threshold(X, spec["bearing"], spec["X_faulty"], w, gamma, _rng(*spec["seed"], 1),
                                      spec["inner_restarts"])
    progress(0.5)
    qubo, blocks = ocsvm_qubo(l1_kernel(X, X, gamma)), [Block(len(X), w)]
    reference = heuristic_reference(qubo, blocks, _rng(*spec["seed"], 2), spec["restarts"])
    progress(1.0)
    return {"tau": tau, "validation_balanced_accuracy": validation, "reference": reference}


def _model_row(X, bits, gamma, tau, tests, qubo, reference, base: dict, stage: str, fixed_point) -> dict:
    """Metrics of the binary one-class SVM whose support vectors are the windows with ``bits == 1``."""
    model = BinaryOCSVM.from_bits(X, bits, gamma)
    energy = float(qubo.energy(bits))
    row = {**base, "stage": stage, "gap": reference.gap(energy), "best_known": reference.is_best(energy),
           **evaluate(model.score, tau, tests)}
    every = np.vstack([tests["healthy"], tests["real"]])
    in_bits, out_bits, exp_bits = fixed_point
    row["fixed_agreement"] = float(np.mean((model.score(every) < tau)
                                           == (model.score_fixed(every, in_bits, out_bits, exp_bits)
                                               < model.fixed_threshold(tau, out_bits))))
    return row


def pce_task(spec: dict, progress) -> dict:
    """One PCE run on the one-class SVM QUBO, evaluated before and after the final local search.

    Spec keys: ``keys`` (arm, round, seed), ``X_train``, ``w``, ``gamma``,
    ``pce`` (:class:`qopt.pce.PCEConfig`), ``seed``, ``reference``, ``tau``,
    ``tests``, ``fixed_point``.
    """
    X, w, gamma = spec["X_train"], spec["w"], spec["gamma"]
    qubo, blocks = ocsvm_qubo(l1_kernel(X, X, gamma)), [Block(len(X), w)]
    res = run_pce(qubo, blocks, spec["pce"], seed=spec["seed"], reference=spec["reference"], progress=progress)
    base = {**spec["keys"], "method": spec["keys"]["arm"]}
    rows = [_model_row(X, res.raw_bits, gamma, spec["tau"], spec["tests"], qubo, spec["reference"], base, "decoded",
                       spec["fixed_point"]),
            _model_row(X, res.bits, gamma, spec["tau"], spec["tests"], qubo, spec["reference"], base, "search",
                       spec["fixed_point"])]
    run = {**spec["keys"], "w": w, "gamma": gamma, "n_train": len(X), **res.summary()}
    return {"rows": rows, "run": run, "params": res.params,
            "bits": {"decoded": np.asarray(res.raw_bits, dtype=np.int8), "search": np.asarray(res.bits, dtype=np.int8)},
            "history": [{**spec["keys"], **h} for h in res.history]}


def random_ocsvm_task(spec: dict, progress) -> list[dict]:
    """Random choices of the ``w`` support vectors, evaluated before and after the same local search as PCE.

    Spec keys: ``keys`` (round), ``X_train``, ``w``, ``gamma``, ``reference``,
    ``tau``, ``tests``, ``draws``, ``search_rounds``, ``seed``, ``fixed_point``.
    """
    X, w, gamma = spec["X_train"], spec["w"], spec["gamma"]
    qubo, blocks = ocsvm_qubo(l1_kernel(X, X, gamma)), [Block(len(X), w)]
    rng = _rng(*spec["seed"])
    rows = []
    for d in range(spec["draws"]):
        bits = random_feasible(blocks, rng)
        searched, _, _ = swap_search(qubo, bits, blocks, spec["search_rounds"])
        base = {**spec["keys"], "method": "random", "arm": "random", "seed": d}
        rows.append(_model_row(X, bits, gamma, spec["tau"], spec["tests"], qubo, spec["reference"], base, "decoded",
                               spec["fixed_point"]))
        rows.append(_model_row(X, searched, gamma, spec["tau"], spec["tests"], qubo, spec["reference"], base,
                               "search", spec["fixed_point"]))
        progress((d + 1) / spec["draws"])
    return rows


def benchmark_task(spec: dict, progress) -> float:
    """Seconds per exact gradient step of one PCE circuit (run in every worker at once to measure the load).

    Spec keys: ``n``, ``layers``, ``m``, ``repeats``, ``ansatz``.
    """
    from qopt.pce import StateVectorCircuit, pauli_strings
    circuit = StateVectorCircuit(spec["n"], spec["layers"], pauli_strings(spec["m"], 2, spec["n"]), spec["ansatz"])
    rng = np.random.default_rng(0)
    params, w = rng.uniform(-np.pi, np.pi, circuit.n_params), rng.normal(size=spec["m"])
    loss = lambda c: (float(w @ c), w)
    circuit.value_and_grad(params, loss)
    t0 = time.perf_counter()
    for _ in range(spec["repeats"]):
        circuit.value_and_grad(params, loss)
    progress(1.0)
    return (time.perf_counter() - t0) / spec["repeats"]
