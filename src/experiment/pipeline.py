"""The experiment pipeline: stages, the tasks of each stage, and the files they write.

Stages (in order; each one reads what the previous ones wrote, so they can
also be run separately, and every stage skips the work already saved):

1. ``data`` -- windows and the 16 features of every bearing (cached per
   bearing in ``PADERBORN/features``).
2. ``costs`` -- ESP32-C6 cycles of every feature and of the FFT
   (``feature_costs.csv``, ``fft_stage.csv``).
3. ``subsets`` -- every subset of ``n_select`` features (4368 for 5 of 16)
   scored, in every round, by scikit-learn's one-class SVM on the test sets
   (``subset_metrics.csv``). Any selection made later (VQE, random, exact) is
   evaluated by looking it up here.
4. ``selection`` -- the selection QUBO of every round (mRMR under the cycle
   budget, penalty weights tuned by enumeration): constrained and
   unconstrained exact optima (``selection_problems.csv``), VQE runs with
   every optimizer and seed (``vqe_runs.csv``, ``vqe_history.csv``) and
   random subsets (``random_selections.csv``).
5. ``ocsvm`` -- the one-class SVM with binary multipliers on the
   constrained exact selection: supervised threshold and best-known
   solution (``ocsvm_prep.json``), PCE runs (``ocsvm_runs.csv``,
   ``loss_history.csv``, ``params/``) and random choices of the support
   vectors, all evaluated before and after the final local search
   (``ocsvm_results.csv``).
6. ``report`` -- tables and figures (:mod:`experiment.report`).

Seeds never depend on the optimizer, so every optimizer of the same problem
and seed starts from the same parameters (paired comparison).
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import cached_property

import numpy as np
import pandas as pd

from pdm.costs import CLOCK_HZ, cost_table, fft_table
from pdm.features import FEATURE_NAMES
from pdm.ocsvm import auto_gamma
from pdm.paderborn import codes
from pdm.protocol import Round, annotate, calibration_rows, make_rounds, test_sets, training_rows
from pdm.selection import all_subsets, redundancy, relevance, subset_mask, tune_penalty
from qopt.feasible import Reference

from . import tasks as T
from .config import STAGES, Config
from .estimates import LARGE_CIRCUIT, MEASUREMENT_SECONDS, Speeds, calibrate, qubits
from .progress import Progress, Runner, Task
from .storage import CSVLog, read_csv, read_json, write_json

log = logging.getLogger("experiment")

#: Expected Adam steps of the 16-qubit PCE circuit (progress bar only).
ADAM_STEPS = 1000
#: Seconds to score one feature subset with scikit-learn (progress bar only).
SUBSET_SECONDS = 0.08
#: Seconds of one VQE iteration (progress bar only).
VQE_STEP_SECONDS = 0.004
#: Subsets per scoring task.
SUBSET_CHUNK = 182


class StageError(RuntimeError):
    """Some tasks of a stage failed; the pipeline stops before the stages that need them."""


class RoundData:
    """Windows of the main condition, scaled with the healthy training windows of one round.

    Args:
        windows: Window table (index ``0..N-1``).
        rnd: The round.
        cfg: Configuration.
    """

    def __init__(self, windows: pd.DataFrame, rnd: Round, cfg: Config):
        self.windows, self.rnd, self.cfg = windows, rnd, cfg
        self.train = training_rows(windows, rnd, cfg.max_train_per_bearing, cfg.seed)
        X = windows[FEATURE_NAMES].to_numpy(dtype=float)
        lo, hi = X[self.train].min(axis=0), X[self.train].max(axis=0)
        self.X = (X - lo) / np.where(hi > lo, hi - lo, 1.0)             # min-max scaling on the training windows
        self.bearing = windows["bearing"].to_numpy()

    @property
    def X_train(self) -> np.ndarray:
        return self.X[self.train]

    @property
    def bearing_train(self) -> np.ndarray:
        return self.bearing[self.train]

    def tests(self, cols=None) -> dict[str, np.ndarray]:
        """Scaled test windows (all features or the columns ``cols``): healthy test bearings and real damage."""
        cols = list(range(len(FEATURE_NAMES))) if cols is None else list(cols)
        return {name: self.X[np.ix_(np.asarray(rows, dtype=int), cols)]
                for name, rows in test_sets(self.windows, self.rnd).items()}

    def calibration(self, cols=None) -> np.ndarray:
        """Scaled artificial-damage windows (all features or the columns ``cols``) that calibrate the threshold."""
        cols = list(range(len(FEATURE_NAMES))) if cols is None else list(cols)
        return self.X[np.ix_(calibration_rows(self.windows), cols)]


def reference_from(d: dict) -> Reference:
    """Rebuild a :class:`qopt.feasible.Reference` saved in ``ocsvm_prep.json``."""
    return Reference(d["kind"], d["energy"], np.asarray(d["bits"], dtype=np.int8), d["random_mean"], d["e_max"],
                     d["n_feasible"], None, d["evaluations"])


class Pipeline:
    """Runs the stages of one profile.

    Args:
        cfg: Configuration.
        runner: Task runner (process pool and progress bar).
    """

    def __init__(self, cfg: Config, runner: Runner):
        self.cfg, self.runner = cfg, runner
        self.out = cfg.results_dir                    # created by run(); plan() only reads it
        self._windows: pd.DataFrame | None = None
        self._rounds: dict[int, RoundData] = {}
        self._problems: dict[int, tuple] = {}
        self._speeds: Speeds | None = None
        self.history = CSVLog(self.out / "loss_history.csv", ["arm", "round", "seed", "iteration"])

    @property
    def progress(self) -> Progress:
        return self.runner.progress

    @property
    def rounds(self) -> list[Round]:
        return [r for r in make_rounds() if self.cfg.rounds is None or r.index in self.cfg.rounds]

    def n_train(self) -> int:
        per_bearing = 4 * len(self.cfg.measurements)
        if self.cfg.max_train_per_bearing is not None:
            per_bearing = min(per_bearing, self.cfg.max_train_per_bearing)
        return 4 * per_bearing

    # ======================================================================
    # Time estimates (progress bar and --plan)
    # ======================================================================

    @property
    def speeds(self) -> Speeds:
        """Seconds per exact gradient and per loss evaluation of the PCE circuit, measured once."""
        if self._speeds is None:
            m, pce = self.n_train(), self.cfg.pce
            n = qubits(m, pce)
            self._speeds = calibrate({(m, n, pce.layers)}, ansatz=pce.ansatz)
            if self.runner.n_jobs > 1 and n >= LARGE_CIRCUIT:
                times = []
                spec = {"n": n, "layers": pce.layers, "m": m, "repeats": 3, "ansatz": pce.ansatz}
                tasks = [Task(f"calibration/{i}", T.benchmark_task, spec, 1.0, times.append)
                         for i in range(self.runner.n_jobs)]
                if not self.runner.run("calibration", tasks) and times:
                    self._speeds.load = max(1.0, float(np.mean(times)) / self._speeds.grad[(n, pce.layers)])
                log.info(f"calibration: {n}-qubit circuits run {self._speeds.load:.1f}x slower with "
                         f"{self.runner.n_jobs} workers busy (effective parallelism "
                         f"{self.runner.n_jobs / self._speeds.load:.1f})")
        return self._speeds

    def pce_seconds(self, arm: str) -> float:
        cfg = self.cfg.pce_config(arm)
        m = self.n_train()
        n = qubits(m, cfg)
        if arm == "pce_adam":
            return min(ADAM_STEPS, cfg.max_steps) * self.speeds.get("grad", n, cfg.layers) + 1.0
        return cfg.spsa_iterations * 2.1 * self.speeds.get("eval", n, cfg.layers) + 1.0

    def vqe_seconds(self, arm: str) -> float:
        v = self.cfg.vqe_config(arm)
        steps = {"adam": min(v.max_steps, ADAM_STEPS), "spsa": 2 * v.spsa_iterations, "cobyla": v.cobyla_maxiter}
        return steps[arm] * VQE_STEP_SECONDS + 0.5

    def wall(self, weights: list[float]) -> float:
        """Wall time of a stage: its CPU time spread over the workers, or its longest task."""
        if not weights:
            return 0.0
        return max(sum(weights) / max(self.runner.n_jobs, 1), max(weights))

    def run_tasks(self, stage: str, tasks: list[Task]) -> None:
        """Run the tasks of a stage, with weights rescaled to the stage's estimated wall time.

        Raises:
            StageError: If any task failed (after all the others finished).
        """
        total = sum(t.weight for t in tasks)
        if total > 0:
            scale = self.wall([t.weight for t in tasks]) / total
            for t in tasks:
                t.weight *= scale
        if tasks:
            log.info(f"{stage}: {len(tasks)} tasks, estimated {fmt_seconds(self.wall([t.weight for t in tasks]))}")
        failed = self.runner.run(stage, tasks)
        if failed:
            raise StageError(f"{failed} of {len(tasks)} tasks of stage '{stage}' failed (details in run.log)")

    def plan(self, stages=STAGES) -> pd.DataFrame:
        """Tasks and estimated wall time of the given stages (before skipping the work already done).

        Estimating the ``ocsvm`` stage times the 16-qubit circuit (see
        :attr:`speeds`), so it is done only when that stage is planned.
        """
        cfg, rows = self.cfg, []
        n_rounds = len(self.rounds)
        data = [MEASUREMENT_SECONDS * len(cfg.measurements)] * len(codes())
        rows.append(("data", len(data), data))
        rows.append(("costs", 1, [0.1]))
        n_subsets = len(all_subsets(len(FEATURE_NAMES), cfg.n_select))
        sub = [SUBSET_SECONDS * SUBSET_CHUNK] * (n_rounds * -(-n_subsets // SUBSET_CHUNK))
        rows.append(("subsets", len(sub), sub))
        sel = [self.vqe_seconds(a) for a in cfg.vqe_arms for _ in range(n_rounds * cfg.vqe_seeds)]
        rows.append(("selection", len(sel), sel))
        if "ocsvm" in stages:
            oc = [30.0] * n_rounds + [0.4 * cfg.random_ocsvm] * n_rounds
            oc += [self.pce_seconds(a) for a in cfg.pce_arms for _ in range(n_rounds * cfg.ocsvm_seeds)]
            rows.append(("ocsvm", len(oc), oc))
        rows.append(("report", 1, [5.0]))
        return pd.DataFrame([{"stage": s, "tasks": n, "cpu_seconds": sum(w), "wall_seconds": self.wall(w)}
                             for s, n, w in rows if s in stages])

    # ======================================================================
    # Data
    # ======================================================================

    def _cache_path(self, bearing: str):
        sig = json.dumps({**self.cfg.features.signature(), "measurements": list(self.cfg.measurements)},
                         sort_keys=True)
        digest = hashlib.sha1(sig.encode()).hexdigest()[:10]
        return self.cfg.data_dir / "features" / self.cfg.condition / f"{bearing}_{digest}.csv"

    def stage_data(self) -> None:
        tasks = []
        for b in codes():
            path = self._cache_path(b)
            if path.exists():
                continue
            spec = {"archives_dir": self.cfg.data_dir / "archives", "raw_dir": self.cfg.data_dir / "raw",
                    "bearing": b, "condition": self.cfg.condition, "measurements": self.cfg.measurements,
                    "features": self.cfg.features}

            def done(rows, path=path, b=b):
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(path, index=False)
                log.info(f"features {b}: {len(rows)} windows")

            tasks.append(Task(f"features/{b}", T.features_task, spec, MEASUREMENT_SECONDS * len(self.cfg.measurements),
                              done))
        self.run_tasks("data", tasks)
        w = self.windows()
        w.drop(columns=FEATURE_NAMES).to_csv(self.out / "windows.csv", index=False)
        counts = w.groupby("group").size().to_dict()
        log.info(f"{self.cfg.condition}: {len(w)} windows ({', '.join(f'{k} {v}' for k, v in counts.items())})")

    def windows(self) -> pd.DataFrame:
        """Window table of the main condition (from the feature cache), with the bearing metadata."""
        if self._windows is None:
            parts = []
            for b in codes():
                path = self._cache_path(b)
                if not path.exists():
                    raise FileNotFoundError(f"features of {b} not found; run the 'data' stage")
                parts.append(pd.read_csv(path))
            self._windows = annotate(pd.concat(parts, ignore_index=True))
        return self._windows

    def round_data(self, index: int) -> RoundData:
        if index not in self._rounds:
            self._rounds[index] = RoundData(self.windows(), make_rounds()[index], self.cfg)
        return self._rounds[index]

    # ======================================================================
    # Costs
    # ======================================================================

    @cached_property
    def costs(self) -> pd.DataFrame:
        return cost_table(self.cfg.features)

    def stage_costs(self) -> None:
        self.costs.to_csv(self.out / "feature_costs.csv", index=False)
        fft_table(self.cfg.features).to_csv(self.out / "fft_stage.csv", index=False)
        t = self.costs
        log.info(f"costs: all 16 features = {t.attrs['total_cycles']:.3g} cycles per window "
                 f"({t.attrs['total_cycles'] / CLOCK_HZ * 1e3:.0f} ms at {CLOCK_HZ / 1e6:.0f} MHz); FFT stage "
                 f"{t.attrs['fft_cycles'] / t.attrs['total_cycles']:.0%}")

    # ======================================================================
    # Every subset, scored by scikit-learn's one-class SVM
    # ======================================================================

    def stage_subsets(self) -> None:
        cfg = self.cfg
        out = CSVLog(self.out / "subset_metrics.csv", ["round", "mask"])
        masks = [subset_mask(x) for x in all_subsets(len(FEATURE_NAMES), cfg.n_select)]
        tasks = []
        for rnd in self.rounds:
            todo = [m for m in masks if not out.done({"round": rnd.index, "mask": m})]
            if not todo:
                continue
            rd = self.round_data(rnd.index)
            for c in range(0, len(todo), SUBSET_CHUNK):
                chunk = todo[c:c + SUBSET_CHUNK]
                spec = {"round": rnd.index, "X_train": rd.X_train, "bearing": rd.bearing_train,
                        "X_faulty": rd.calibration(), "tests": rd.tests(), "masks": chunk, "nu": cfg.nu}
                tasks.append(Task(f"subsets/r{rnd.index}/{c}", T.subsets_task, spec, SUBSET_SECONDS * len(chunk),
                                  out.append))
        self.run_tasks("subsets", tasks)

    def subset_metrics(self) -> pd.DataFrame:
        df = read_csv(self.out / "subset_metrics.csv")
        if df.empty:
            raise FileNotFoundError("subset_metrics.csv not found; run the 'subsets' stage")
        return df

    # ======================================================================
    # Feature selection: exact optimum, VQE and random subsets
    # ======================================================================

    def problem(self, rnd: int):
        """The selection QUBO of a round (mRMR under the cycle budget) and the summary of its penalty tuning.

        The relevance compares the healthy training windows with the
        artificial-damage windows; the redundancy uses the healthy training
        windows. The penalty weights are tuned by enumeration
        (:func:`pdm.selection.tune_penalty`).
        """
        if rnd not in self._problems:
            rd = self.round_data(rnd)
            a = relevance(rd.X_train, rd.calibration(), seed=self.cfg.seed + rnd)
            self._problems[rnd] = tune_penalty(a, redundancy(rd.X_train), self.costs, self.cfg.n_select,
                                               self.cfg.cycle_budget)
        return self._problems[rnd]

    def stage_selection(self) -> None:
        cfg = self.cfg
        problems_log = CSVLog(self.out / "selection_problems.csv", ["round"])
        runs = CSVLog(self.out / "vqe_runs.csv", ["arm", "round", "seed"])
        history = CSVLog(self.out / "vqe_history.csv", ["arm", "round", "seed", "iteration"])
        randoms = CSVLog(self.out / "random_selections.csv", ["round", "draw"])
        tasks = []
        for rnd in self.rounds:
            r = rnd.index
            n = len(FEATURE_NAMES)
            if not randoms.done({"round": r, "draw": 0}):
                rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 71, r]))
                rows = []
                for d in range(cfg.random_selections):
                    x = np.zeros(n, dtype=np.int8)
                    x[rng.choice(n, cfg.n_select, replace=False)] = 1
                    rows.append({"round": r, "draw": d, "mask": subset_mask(x)})
                randoms.append(rows)
            problem, info = self.problem(r)
            x, parts = problem.enumerate()
            c, u = problem.optimum()
            # The gap is measured on the mRMR objective: 0 = constrained optimum, 1 = mean of all subsets.
            optimum, mean = float(parts["objective"].iloc[c]), float(parts["objective"].mean())
            if not problems_log.done({"round": r}):
                problems_log.append({
                    "round": r, "budget": cfg.cycle_budget, **info, "link": problem.link,
                    "mask": subset_mask(x[c]), "features": json.dumps(problem.selected(x[c])),
                    "objective": float(parts["objective"].iloc[c]), "cycles": float(parts["cycles"].iloc[c]),
                    "energy": float(parts["energy"].iloc[c]), "objective_mean": mean,
                    "free_mask": subset_mask(x[u]), "free_features": json.dumps(problem.selected(x[u])),
                    "free_objective": float(parts["objective"].iloc[u]), "free_cycles": float(parts["cycles"].iloc[u]),
                    "feasible_share": float(parts["feasible"].mean()),
                    "relevance": json.dumps(dict(zip(problem.names, np.round(problem.a, 4).tolist())))})
                log.info(f"selection r{r}: optimum {problem.selected(x[c])}, {parts['cycles'].iloc[c] / 1e6:.1f} M "
                         f"cycles (budget {cfg.cycle_budget / 1e6:.0f} M; without it: {problem.selected(x[u])}); "
                         f"lambda = ({info['lambda_1']:.3g}, {info['lambda_2']:.3g}), {info['infeasible_below']} "
                         f"infeasible subsets below the optimum")
            for arm in cfg.vqe_arms:
                for seed in range(cfg.vqe_seeds):
                    keys = {"arm": arm, "round": r, "seed": seed}
                    if runs.done(keys):
                        continue
                    spec = {"keys": keys, "problem": problem, "vqe": cfg.vqe_config(arm),
                            "seed": [cfg.seed, 51, r, seed], "optimum_mask": subset_mask(x[c]),
                            "optimum": optimum, "random_mean": mean}

                    def done(result, keys=keys):
                        history.append(result["history"])
                        runs.append(result["row"])
                        row = result["row"]
                        log.debug(f"VQE {keys}: gap {row['gap']:.3f}, optimum {row['is_optimum']}, "
                                  f"feasible {row['feasible']}")

                    tasks.append(Task(f"vqe/{arm}/r{r}/s{seed}", T.vqe_task, spec, self.vqe_seconds(arm), done))
        self.run_tasks("selection", tasks)
        df = read_csv(self.out / "vqe_runs.csv")
        if not df.empty:
            summary = df.groupby("arm").agg(optimum=("is_optimum", "mean"), gap=("gap", "mean"))
            log.info("VQE: " + "; ".join(f"{a}: optimum {r.optimum:.0%}, mean gap {r.gap:.3f}"
                                         for a, r in summary.iterrows()))

    # ======================================================================
    # One-class SVM with binary multipliers (PCE)
    # ======================================================================

    def ocsvm_features(self, rnd: int) -> list[str]:
        """Features of the one-class SVM of a round: the constrained exact selection."""
        probs = read_csv(self.out / "selection_problems.csv")
        if probs.empty:
            raise FileNotFoundError("selection_problems.csv not found; run the 'selection' stage")
        row = probs[probs["round"] == rnd].iloc[0]
        return json.loads(row["features"])

    def stage_ocsvm(self) -> None:
        cfg = self.cfg
        prep_path = self.out / "ocsvm_prep.json"
        prep = read_json(prep_path, {}) or {}
        runs = CSVLog(self.out / "ocsvm_runs.csv", ["arm", "round", "seed"])
        results = CSVLog(self.out / "ocsvm_results.csv", ["method", "round", "seed", "stage"])
        inputs = {}
        prep_tasks = []
        for rnd in self.rounds:
            r = rnd.index
            rd = self.round_data(r)
            feats = self.ocsvm_features(r)
            cols = [FEATURE_NAMES.index(f) for f in feats]
            X = rd.X_train[:, cols]
            gamma = auto_gamma(X)
            inputs[r] = {"X": X, "gamma": gamma, "tests": rd.tests(cols), "features": feats}
            if str(r) in prep:
                continue
            spec = {"X_train": X, "bearing": rd.bearing_train, "X_faulty": rd.calibration(cols), "w": cfg.w,
                    "gamma": gamma, "inner_restarts": cfg.inner_restarts,
                    "restarts": cfg.reference_restarts, "seed": [cfg.seed, 91, r]}

            def done(result, r=r, feats=feats):
                ref = result["reference"]
                prep[str(r)] = {"tau": result["tau"], "validation_balanced_accuracy": result["validation_balanced_accuracy"],
                                "features": feats,
                                "reference": {"kind": ref.kind, "energy": ref.energy, "bits": ref.bits.tolist(),
                                              "random_mean": ref.random_mean, "e_max": ref.e_max,
                                              "n_feasible": ref.n_feasible, "evaluations": ref.evaluations}}
                write_json(prep_path, prep)
                log.info(f"OCSVM r{r}: features {feats}, threshold {result['tau']:.4f} (validation balanced "
                         f"accuracy {result['validation_balanced_accuracy']:.3f})")

            prep_tasks.append(Task(f"ocsvm_prep/r{r}", T.ocsvm_prep_task, spec, 30.0, done))
        self.run_tasks("ocsvm (thresholds and references)", prep_tasks)

        tasks = []
        for rnd in self.rounds:
            r = rnd.index
            inp, p = inputs[r], prep[str(r)]
            reference = reference_from(p["reference"])
            common = {"X_train": inp["X"], "w": cfg.w, "gamma": inp["gamma"], "reference": reference,
                      "tau": p["tau"], "tests": inp["tests"], "fixed_point": cfg.fixed_point}
            if not results.done({"method": "random", "round": r, "seed": 0, "stage": "search"}):
                spec = {**common, "keys": {"round": r}, "draws": cfg.random_ocsvm,
                        "search_rounds": cfg.pce.search_rounds, "seed": [cfg.seed, 81, r]}
                tasks.append(Task(f"ocsvm_random/r{r}", T.random_ocsvm_task, spec, 0.4 * cfg.random_ocsvm,
                                  results.append))
            for arm in cfg.pce_arms:
                for seed in range(cfg.ocsvm_seeds):
                    keys = {"arm": arm, "round": r, "seed": seed}
                    if results.done({"method": arm, "round": r, "seed": seed, "stage": "search"}):
                        continue
                    spec = {**common, "keys": keys, "pce": cfg.pce_config(arm), "seed": [cfg.seed, 61, r, seed]}

                    def done(result, keys=keys):
                        self.history.append(result["history"])
                        stem = f"ocsvm_{keys['arm']}_r{keys['round']}_s{keys['seed']}"
                        np.save(self.out / "params" / f"{stem}.npy", result["params"])
                        np.savez(self.out / "params" / f"{stem}_bits.npz", **result["bits"])   # re-evaluation
                        runs.append(result["run"])
                        results.append(result["rows"])
                        raw, final = result["rows"]
                        log.info(f"OCSVM {keys['arm']} r{keys['round']} s{keys['seed']}: decoded gap "
                                 f"{raw['gap']:.3f} (AUC {raw['auc']:.3f}), after search {final['gap']:.3f} "
                                 f"(AUC {final['auc']:.3f})")

                    tasks.append(Task(f"ocsvm/{arm}/r{r}/s{seed}", T.pce_task, spec, self.pce_seconds(arm), done))
        self.run_tasks("ocsvm", tasks)

    # ======================================================================
    # Report
    # ======================================================================

    def stage_report(self) -> None:
        from .report import make_report
        make_report(self)

    def run(self, stages: list[str]) -> None:
        """Run the given stages in pipeline order."""
        (self.out / "params").mkdir(parents=True, exist_ok=True)
        todo = [s for s in STAGES if s in stages]
        for i, stage in enumerate(todo):
            log.info(f"=== stage: {stage} ===")
            getattr(self, f"stage_{stage}")()
            remaining = set(todo[i + 1:])
            if remaining:
                for _, row in self.plan(remaining).iterrows():
                    self.progress.plan(row["stage"], row["wall_seconds"])


def fmt_seconds(seconds: float) -> str:
    """``1h 02m``, ``5m 30s`` or ``12s``."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")
