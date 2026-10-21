"""Figures of the article, drawn from the CSV files of a run.

* :func:`selection_histograms` -- feature selection (Figure 2): 2D histograms
  of the subsets chosen by each method, cycles per window (x) against the
  normalized mRMR objective or the detection quality of scikit-learn's
  one-class SVM with those features (y).
* :func:`decoded_gaps` -- PCE (Figure 3): gap of the decoded solutions to the
  best-known solution, before local search.
* :func:`loss_curves` -- PCE (Figure 4): exact loss vs. iteration.
* :func:`ocsvm_violins` -- one-class SVM with binary multipliers (Figure 5):
  violins of accuracy, F1 and AUC of the detectors trained by each method,
  before and/or after the final local search.

Figure 3 fits one column of the IEEE template (``style.SINGLE``); the others
span both columns (``style.DOUBLE``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # no display needed (figures are saved to files)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import style  # noqa: E402
from .config import LABELS  # noqa: E402

#: Color of each method (a color follows an entity across figures).
COLORS = {"adam": style.CATEGORICAL[0], "spsa": style.CATEGORICAL[1], "cobyla": style.CATEGORICAL[2],
          "random": style.TEXT_2, "pce_adam": style.CATEGORICAL[0], "pce_spsa": style.CATEGORICAL[1]}
METRIC_LABELS = {"f1": "F1 score", "auc": "AUC", "gap": "mRMR objective\n(0: optimum; 1: random)",
                 "accuracy": "accuracy", "balanced_accuracy": "balanced accuracy"}


def close(fig) -> None:
    plt.close(fig)


def selection_histograms(points: pd.DataFrame, optima: pd.DataFrame, metric, path: Path | None = None,
                         budget: float | None = None):
    """2D histograms of the selected subsets: cycles per window against one or more metrics.

    One column per method (VQE optimizers and random) and one row per metric,
    all rows sharing the cycles axis.

    Args:
        points: One row per selection with ``method``, ``cycles`` and the metrics.
        optima: Exact optimum of every round (``cycles`` and the metrics), drawn as markers.
        metric: Column of the y axis (``f1``, ``auc``, ``gap`` ...), or a tuple of columns (one row each).
        budget: Cycle budget per window, drawn as a vertical dashed line (``None``: no line).
    """
    style.apply()
    metrics = (metric,) if isinstance(metric, str) else tuple(metric)
    methods = [m for m in ("adam", "spsa", "cobyla", "random") if m in set(points["method"])]
    height = 2.25 if len(metrics) == 1 else 1.75 * len(metrics) + 0.45
    fig, axes = plt.subplots(len(metrics), len(methods), figsize=(style.DOUBLE, height), sharex=True,
                             sharey="row", squeeze=False)
    x_all = points["cycles"] / 1e6
    x_bins = np.linspace(x_all.min() - 0.5, x_all.max() + 0.5, 16)
    image = None
    for row, name in enumerate(metrics):
        y_all = points[name]
        y_lo, y_hi = float(np.nanmin(y_all)), float(np.nanmax(y_all))
        pad = 0.02 * max(y_hi - y_lo, 1e-3)
        y_bins = np.linspace(y_lo - pad, y_hi + pad, 14)
        for ax, method in zip(axes[row], methods):
            sub = points[points["method"] == method]
            h, _, _ = np.histogram2d(sub["cycles"] / 1e6, sub[name], bins=[x_bins, y_bins])
            h = h / max(h.sum(), 1)                               # fraction of the selections of this method
            h[h == 0] = np.nan
            image = ax.pcolormesh(x_bins, y_bins, h.T, cmap=style.SEQ, vmin=0, vmax=0.25, shading="flat")
            ax.scatter(optima["cycles"] / 1e6, optima[name], marker="*", s=28, color=style.CATEGORICAL[3],
                       edgecolors=style.TEXT, linewidths=0.4, zorder=3, label="exact optimum")
            if budget is not None:
                ax.axvline(budget / 1e6, color=style.TEXT_2, linewidth=0.7, linestyle="--", zorder=2,
                           label="cycle budget")
            if row == 0:
                ax.set_title(LABELS.get(method, method), fontsize=7)
            ax.grid(True, linewidth=0.4)
        axes[row, 0].set_ylabel(METRIC_LABELS.get(name, name), fontsize=7 if len(metrics) > 1 else None)
    axes[-1, 0].legend(loc="upper right" if metrics[-1] == "gap" else "lower right", fontsize=6)
    fig.supxlabel("cycles per window on the ESP32-C6 (millions)", fontsize=8, y=0.02)
    bottom = 0.2 if len(metrics) == 1 else 0.11
    top = 0.88 if len(metrics) == 1 else 0.93
    fig.subplots_adjust(left=0.08, right=0.91, bottom=bottom, top=top, wspace=0.08, hspace=0.12)
    cax = fig.add_axes([0.925, bottom, 0.012, top - bottom])
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("fraction of selections", fontsize=6)
    bar.ax.tick_params(labelsize=6)
    if path is not None:
        style.save(fig, path)
    return fig


def ocsvm_violins(results: pd.DataFrame, stages: tuple[str, ...], path: Path | None = None,
                  metrics: tuple[str, ...] = ("accuracy", "f1", "auc")):
    """Violins of the detection metrics of the binary one-class SVMs trained by each method.

    Args:
        results: One row per detector with ``method``, ``stage`` (``decoded`` or
            ``search``) and the metric columns.
        stages: Rows of panels (``("decoded",)``, ``("search",)`` or both).
        metrics: Columns of panels.
    """
    style.apply()
    methods = [m for m in ("pce_adam", "pce_spsa", "random") if m in set(results["method"])]
    stage_titles = {"decoded": "decoded solution (no local search)", "search": "after the final local search"}
    fig, axes = plt.subplots(len(stages), len(metrics), figsize=(style.DOUBLE, 1.9 * len(stages) + 0.3),
                             squeeze=False, sharey="col")
    for i, stage in enumerate(stages):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            data = [results[(results["method"] == m) & (results["stage"] == stage)][metric].dropna().to_numpy()
                    for m in methods]
            positions = np.arange(len(methods))
            parts = ax.violinplot([d if len(d) else [np.nan] for d in data], positions=positions, widths=0.75,
                                  showmedians=True, showextrema=False)
            for body, m in zip(parts["bodies"], methods):
                body.set_facecolor(COLORS[m])
                body.set_edgecolor(COLORS[m])
                body.set_alpha(0.55)
            parts["cmedians"].set_color(style.TEXT)
            parts["cmedians"].set_linewidth(1.0)
            for p, d, m in zip(positions, data, methods):
                if len(d):
                    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(d))
                    ax.scatter(p + jitter, d, s=4, color=COLORS[m], edgecolors="none", alpha=0.8, zorder=3)
            ax.set_xticks(positions)
            ax.set_xticklabels([LABELS.get(m, m).replace(" (", "\n(") for m in methods], fontsize=6)
            if i == 0:
                ax.set_title(METRIC_LABELS.get(metric, metric))
            if j == 0:
                ax.set_ylabel(stage_titles[stage], fontsize=6.5)
    fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.6)
    if path is not None:
        style.save(fig, path)
    return fig


def decoded_gaps(runs: pd.DataFrame, path: Path | None = None):
    """Normalized gap of the decoded PCE solutions (before local search), per optimizer and for the untrained circuit.

    One point per run (all rounds pooled) and a bar at the median; 0 is the
    best-known solution and the dashed line (1) the mean of a random choice of
    the support vectors. The untrained circuit is the same circuit with its
    initial parameters, read and decoded in the same way.

    Args:
        runs: ``ocsvm_runs.csv`` (columns ``arm``, ``raw_gap`` and ``initial_raw_gap``).
    """
    style.apply()
    groups = [(LABELS[a].split(" (")[0].replace("PCE + ", ""), runs.loc[runs["arm"] == a, "raw_gap"], COLORS[a], "o")
              for a in ("pce_adam", "pce_spsa") if a in set(runs["arm"])]
    groups.append(("untrained circuit", runs["initial_raw_gap"], style.TEXT_2, "D"))
    fig, ax = plt.subplots(figsize=(style.SINGLE, 2.2))
    rng = np.random.default_rng(0)
    for i, (label, values, color, marker) in enumerate(groups):
        values = values.dropna().to_numpy()
        if not len(values):
            continue
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(values)), values, s=10, color=color, marker=marker,
                   edgecolors=style.SURFACE, linewidths=0.4, zorder=3)
        median = float(np.median(values))
        ax.plot([i - 0.3, i + 0.3], [median, median], color=style.TEXT, linewidth=1.2, zorder=4)
        ax.text(i + 0.33, median, f"{median:.2f}", ha="left", va="center", fontsize=5.8, color=style.TEXT)
    ax.axhline(1.0, color=style.TEXT_2, linewidth=0.8, linestyle="--", zorder=2)
    ax.axhline(0.0, color=style.TEXT_2, linewidth=0.6, zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=6)
    ax.set_xlim(-0.5, len(groups) - 0.4)
    ax.set_ylabel("normalized gap")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.17)
    if path is not None:
        style.save(fig, path)
    return fig


def loss_curves(history: pd.DataFrame, path: Path | None = None):
    """Exact PCE loss vs. iteration, one panel per round: median (line) and range (band) over the runs of each arm.

    The curves of the runs are put on a common logarithmic grid of iterations
    by linear interpolation; a run that stopped early (Adam's convergence
    criterion) keeps its last value, since its parameters no longer change.
    """
    style.apply()
    history = history.sort_values("iteration")
    rounds = sorted(history["round"].unique())
    grid = np.unique(np.round(np.logspace(0, np.log10(history["iteration"].max()), 200)).astype(int))
    fig, axes = plt.subplots(1, len(rounds), figsize=(style.DOUBLE, 2.1), squeeze=False)
    for ax, r in zip(axes[0], rounds):
        for arm in ("pce_adam", "pce_spsa"):
            runs = history[(history["arm"] == arm) & (history["round"] == r)]
            if runs.empty:
                continue
            curves = np.array([np.interp(grid, g["iteration"].to_numpy(), g["loss"].to_numpy())
                               for _, g in runs.groupby("seed")])
            ax.fill_between(grid, curves.min(axis=0), curves.max(axis=0), color=COLORS[arm], alpha=0.22, linewidth=0)
            ax.plot(grid, np.median(curves, axis=0), color=COLORS[arm], linewidth=1.3, label=LABELS[arm])
        ax.set_xscale("log")
        ax.set_title(f"round {int(r) + 1}", fontsize=7)
        ax.set_xlabel("iteration")
    axes[0, 0].set_ylabel("exact loss")
    axes[0, 0].legend(fontsize=6, loc="upper right")
    fig.subplots_adjust(left=0.105, right=0.99, top=0.88, bottom=0.22, wspace=0.32)
    if path is not None:
        style.save(fig, path)
    return fig
