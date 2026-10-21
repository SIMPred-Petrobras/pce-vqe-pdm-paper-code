"""Summary tables, figures and a Markdown summary of a run (stage ``report``).

Everything here is recomputed from the CSV files of the run, so the report
can be regenerated at any time (``python src/main.py --stages report``). The
files are written to ``<results>/report/``:

* ``selections.csv`` -- every feature selection (VQE runs, random draws and
  exact optima, with and without the cycle budget) with its cycles,
  feasibility, normalized mRMR gap and the detection metrics of
  scikit-learn's one-class SVM with those features;
* ``selection_quality.csv`` and ``selection_quality_by_round.csv`` -- per
  method (and round): optimum rate, feasibility rate, mean gap, cycles, F1
  and AUC;
* ``ocsvm_quality.csv`` -- per method and stage (decoded / after search):
  gap to the best-known solution and detection metrics of the binary
  one-class SVMs;
* figures of the article: ``fig_selection_gap_f1`` (Figure 2),
  ``fig_decoded_gap`` (Figure 3), ``fig_loss_curves`` (Figure 4) and
  ``fig_ocsvm_violins`` (Figure 5); and the variants ``fig_selection_f1``,
  ``fig_selection_auc``, ``fig_ocsvm_violins_decoded`` and
  ``fig_ocsvm_violins_search``;
* ``summary.md`` -- the tables above and the numbers quoted in the article
  (exact optima, thresholds, cost of inference, training cost).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from pdm.costs import CLOCK_HZ, ocsvm_inference_cycles
from pdm.selection import subset_mask
from qopt.pce import mean_spin_bound, pauli_strings, resolve

from . import figures
from .storage import read_csv, read_json

log = logging.getLogger("experiment")

DETECTION = ["accuracy", "balanced_accuracy", "f1", "auc", "recall", "false_alarm_rate"]


def markdown(df: pd.DataFrame, digits: int = 3) -> str:
    """A DataFrame as a Markdown table."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule] + body)


def selections(p) -> pd.DataFrame:
    """Every selection (VQE, random, exact optima) with its QUBO energy, gap, cycles and detection metrics.

    ``exact`` is the constrained mRMR optimum of each round and
    ``exact_free`` the optimum without the cycle budget. ``is_optimum`` means
    "the constrained optimum"; ``feasible`` means "within the cycle budget".
    """
    metrics = p.subset_metrics().set_index(["round", "mask"])
    vqe = read_csv(p.out / "vqe_runs.csv")
    randoms = read_csv(p.out / "random_selections.csv")
    probs = read_csv(p.out / "selection_problems.csv")
    rows = []
    for r in sorted(probs["round"].unique()):
        problem, _ = p.problem(int(r))
        x, parts = problem.enumerate()
        table = parts.assign(mask=[subset_mask(row) for row in x]).set_index("mask")
        prob = probs[probs["round"] == r].iloc[0]
        optimum, mean = float(prob["objective"]), float(prob["objective_mean"])
        span = mean - optimum if mean > optimum else 1.0

        def scored(mask, method, seed):
            t = table.loc[int(mask)]
            return {"method": method, "round": int(r), "seed": seed, "mask": int(mask), "cycles": float(t["cycles"]),
                    "objective": float(t["objective"]), "relevance": float(t["relevance"]),
                    "redundancy": float(t["redundancy"]), "feasible": bool(t["feasible"]),
                    "energy": float(t["energy"]), "gap": (float(t["objective"]) - optimum) / span,
                    "is_optimum": int(mask) == int(prob["mask"])}

        rows.append(scored(prob["mask"], "exact", 0))
        rows.append(scored(prob["free_mask"], "exact_free", 0))
        for _, v in vqe[vqe["round"] == r].iterrows():
            rows.append({**scored(v["mask"], v["arm"], int(v["seed"])), "p_answer": v["p_answer"]})
        for _, d in randoms[randoms["round"] == r].iterrows():
            rows.append(scored(d["mask"], "random", int(d["draw"])))
    df = pd.DataFrame(rows)
    joined = metrics.loc[list(zip(df["round"], df["mask"]))].reset_index(drop=True)
    return pd.concat([df.reset_index(drop=True), joined[["features"] + [c for c in DETECTION if c in joined]]],
                     axis=1)


def _costs_section(p) -> list[str]:
    costs = read_csv(p.out / "feature_costs.csv")
    if costs.empty:
        return []
    total = costs["own_cycles"].sum() + costs["stage_cycles"].max()
    fft = costs["stage_cycles"].max()
    inference = ocsvm_inference_cycles(p.cfg.w, p.cfg.n_select)
    return ["## Cycles per window on the ESP32-C6 (model)", "",
            markdown(costs[["feature", "group", "own_cycles", "standalone_cycles"]].assign(
                share=lambda d: d["own_cycles"] / total), 3), "",
            f"FFT stage: {fft:,.0f} cycles ({fft / total:.0%} of the total); all 16 features: {total:,.0f} cycles "
            f"({total / CLOCK_HZ * 1e3:.0f} ms at {CLOCK_HZ / 1e6:.0f} MHz). Cycle budget: {p.cfg.cycle_budget:,.0f}. "
            f"Integer-only decision of the one-class SVM ({p.cfg.w} support vectors, {p.cfg.n_select} features): "
            f"{inference:,.0f} cycles.", ""]


def _selection_section(p, out) -> list[str]:
    try:
        sel = selections(p)
    except (FileNotFoundError, KeyError) as error:
        log.warning(f"report: feature selection skipped ({error})")
        return []
    if sel.empty:
        return []
    sel.to_csv(out / "selections.csv", index=False)
    q = (sel.groupby("method")
         .agg(selections=("mask", "size"), optimum_rate=("is_optimum", "mean"),
              feasible_rate=("feasible", "mean"), gap_mean=("gap", "mean"),
              cycles_mean=("cycles", "mean"), f1_mean=("f1", "mean"), f1_std=("f1", "std"),
              auc_mean=("auc", "mean"), auc_std=("auc", "std"))
         .reset_index())
    q.to_csv(out / "selection_quality.csv", index=False)
    by_round = (sel.groupby(["method", "round"])
                .agg(optimum_rate=("is_optimum", "mean"), feasible_rate=("feasible", "mean"),
                     gap_mean=("gap", "mean"), cycles_mean=("cycles", "mean"), f1_mean=("f1", "mean"),
                     auc_mean=("auc", "mean")).reset_index())
    by_round.to_csv(out / "selection_quality_by_round.csv", index=False)

    probs = read_csv(p.out / "selection_problems.csv")
    optima = []
    for _, row in probs.iterrows():
        for kind, mask, feats, cycles in (("within budget", row["mask"], row["features"], row["cycles"]),
                                          ("without budget", row["free_mask"], row["free_features"],
                                           row["free_cycles"])):
            m = sel[(sel["round"] == row["round"]) & (sel["mask"] == mask)].iloc[0]
            optima.append({"round": int(row["round"]) + 1, "optimum": kind,
                           "lambda_1, lambda_2": f"{row['lambda_1']:.3g}, {row['lambda_2']:.3g}",
                           "features": ", ".join(json.loads(feats)), "cycles (M)": cycles / 1e6,
                           "objective": m["objective"], "f1": m["f1"], "auc": m["auc"]})
    vqe = read_csv(p.out / "vqe_runs.csv")
    cost = (vqe.groupby("arm").agg(runs=("seed", "size"), steps_mean=("steps", "mean"),
                                   evaluations_mean=("evaluations", "mean"),
                                   training_shots_mean=("training_shots", "mean"), seconds_mean=("seconds", "mean"),
                                   p_answer_mean=("p_answer", "mean")).reset_index())
    lines = ["## Feature selection (VQE vs. random vs. exact; pooled over rounds and seeds)", "",
             "`gap`: normalized mRMR gap (0 = constrained optimum, 1 = mean of all subsets); `feasible`: within the "
             "cycle budget; F1 and AUC: scikit-learn's one-class SVM with the selected features, on the real damage.",
             "", markdown(q), "", "### By round", "", markdown(by_round), "",
             "### Exact optima and penalty weights", "",
             f"Share of the subsets within the budget: {probs['feasible_share'].mean():.0%}; infeasible subsets with "
             f"a lower QUBO energy than the constrained optimum (all rounds): {int(probs['infeasible_below'].sum())}.",
             "", markdown(pd.DataFrame(optima)), "",
             "### Cost of the VQE runs", "", markdown(cost), ""]

    points = sel[~sel["method"].isin(["exact", "exact_free"])]
    exact = sel[sel["method"] == "exact"]
    for metric in ("f1", "auc"):
        figures.close(figures.selection_histograms(points, exact, metric, out / f"fig_selection_{metric}",
                                                   budget=p.cfg.cycle_budget))
    figures.close(figures.selection_histograms(points, exact, ("gap", "f1"), out / "fig_selection_gap_f1",
                                               budget=p.cfg.cycle_budget))
    return lines


def _ocsvm_section(p, out) -> list[str]:
    oc = read_csv(p.out / "ocsvm_results.csv")
    if oc.empty:
        return []
    cfg = p.cfg
    q = (oc.groupby(["method", "stage"])
         .agg(detectors=("auc", "size"), gap_mean=("gap", "mean"), gap_median=("gap", "median"),
              best_known_rate=("best_known", "mean"),
              **{f"{m}_mean": (m, "mean") for m in ("accuracy", "f1", "auc", "balanced_accuracy",
                                                    "false_alarm_rate", "recall")},
              **{f"{m}_std": (m, "std") for m in ("accuracy", "f1", "auc")},
              fixed_agreement=("fixed_agreement", "mean"))
         .reset_index())
    q.to_csv(out / "ocsvm_quality.csv", index=False)
    lines = ["## One-class SVM with binary multipliers", "",
             "`gap`: normalized gap to the best of the classical local searches (0 = best known, 1 = mean of a "
             "random choice); `decoded`: the w largest correlators set to 1; `search`: after one round of local "
             "search; `fixed_agreement`: share of test windows on which the integer-only decision equals the "
             "floating-point one.", "", markdown(q), ""]

    prep = read_json(p.out / "ocsvm_prep.json", {}) or {}
    if prep:
        rows = [{"round": int(r) + 1, "features": ", ".join(v["features"]), "threshold tau": v["tau"],
                 "validation balanced accuracy": v["validation_balanced_accuracy"]} for r, v in sorted(prep.items())]
        lines += ["### Features and supervised thresholds", "", markdown(pd.DataFrame(rows)), ""]

    runs = read_csv(p.out / "ocsvm_runs.csv")
    if not runs.empty:
        cost = (runs.groupby("arm").agg(runs=("seed", "size"), decoded_gap_median=("raw_gap", "median"),
                                        untrained_gap_median=("initial_raw_gap", "median"),
                                        steps_median=("steps", "median"),
                                        loss_evaluations_mean=("loss_evaluations", "mean"),
                                        training_shots_mean=("training_shots_used", "mean"),
                                        hours_mean=("seconds", lambda s: s.mean() / 3600)).reset_index())
        m = int(runs["n_train"].iloc[0])
        n, _, alpha = resolve(m, cfg.pce)
        bound = mean_spin_bound(n, pauli_strings(m, cfg.pce.k, n), alpha)
        lines += ["### PCE runs", "", markdown(cost), "",
                  f"Correlator bound ({m} variables on {n} qubits, alpha = {alpha:g}): the mean of "
                  f"tanh(alpha <Pi_i>) cannot go below {bound:.3f}, while the cardinality penalty would ask for "
                  f"2w/m - 1 = {2 * cfg.w / m - 1:.3f} (see qopt.pce.mean_spin_bound).", ""]
        figures.close(figures.decoded_gaps(runs, out / "fig_decoded_gap"))

    for stages, name in ((("decoded", "search"), "fig_ocsvm_violins"), (("decoded",), "fig_ocsvm_violins_decoded"),
                         (("search",), "fig_ocsvm_violins_search")):
        figures.close(figures.ocsvm_violins(oc, stages, out / name))
    return lines


def make_report(p) -> None:
    """Write the tables, figures and ``summary.md`` of a run."""
    out = p.out / "report"
    out.mkdir(exist_ok=True)
    lines = [f"# Results of profile `{p.cfg.profile}`", ""]
    lines += _costs_section(p)
    lines += _selection_section(p, out)
    lines += _ocsvm_section(p, out)
    history = read_csv(p.out / "loss_history.csv")
    if not history.empty:
        figures.close(figures.loss_curves(history.drop_duplicates(["arm", "round", "seed", "iteration"]),
                                          out / "fig_loss_curves"))
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(f"report written to {out}")
