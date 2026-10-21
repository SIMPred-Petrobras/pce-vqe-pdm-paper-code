"""Matplotlib palette, style and figure sizes shared by the figures.

Sizes follow the IEEEcsmag template: two columns of about 3.5 in and a full
text width of about 7 in. Figures are drawn at their final size with 7-8 pt
fonts, so they can be included with ``width=\\columnwidth`` (``SINGLE``) or
``width=\\textwidth`` in a ``figure*`` (``DOUBLE``) without rescaling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SINGLE = 3.5   # width (in) of one column
DOUBLE = 7.0   # width (in) of both columns (figure*)

# Sequential ramp (magnitude): a single hue, from light to dark.
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)

# Categorical colors in a fixed order; a color follows an entity, never a position.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def apply() -> None:
    """Set the global matplotlib style used by every figure."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.edgecolor": GRID, "axes.linewidth": 0.6,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 4,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "legend.frameon": False, "legend.handlelength": 1.5,
        "lines.linewidth": 1.5, "lines.markersize": 4, "errorbar.capsize": 2,
    })


def save(fig, path: str | Path) -> None:
    """Save a figure as PDF (vector, for LaTeX) and PNG, at its current size."""
    path = Path(path)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))

