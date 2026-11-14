"""Small presentation helpers for the anomaly walkthrough."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

COLORS = {"signal": "#286C63", "spike": "#B85835", "change": "#8D6AA3", "motif": "#387AB2"}
SOURCE = "Synthetic one-minute telemetry · SQL construction shown above · real vgi-anomaly worker / Query.Farm"


def validate_parameters(window: int, threshold: float) -> None:
    """Validate the bounded interactive controls before starting a worker."""
    if isinstance(window, bool) or not isinstance(window, int) or not 6 <= window <= 60:
        raise ValueError("WINDOW_SAMPLES must be an integer from 6 to 60 (one sample per minute).")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise ValueError("ZSCORE_THRESHOLD must be a number from 1.5 to 5.0.")
    if not math.isfinite(threshold) or not 1.5 <= threshold <= 5.0:
        raise ValueError("ZSCORE_THRESHOLD must be finite and from 1.5 to 5.0.")


def prose(text: str) -> None:
    """Display a result-derived interpretation with folded source."""
    display(Markdown(text))


def set_theme() -> None:
    """Apply the common notebook visual style."""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "figure.dpi": 140,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CCD5CF",
            "axes.labelcolor": "#405047",
            "text.color": "#243B2F",
            "grid.color": "#E8EDE9",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "legend.frameon": False,
            "lines.linewidth": 1.7,
            "text.parse_math": False,
        },
    )
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.precision", 3)


def finish_chart(fig: Any, title: str, subtitle: str) -> None:
    """Give a figure space for the question, measurement context and provenance."""
    fig.text(0.075, 0.985, title, va="top", fontsize=17, fontweight="semibold")
    fig.text(0.075, 0.925, subtitle, va="top", fontsize=10.5, color="#58685F")
    fig.tight_layout(rect=(0.01, 0.07, 0.99, 0.875), h_pad=2.0, w_pad=2.4)
    fig.text(0.075, 0.018, SOURCE, va="bottom", fontsize=8.2, color="#78857D")
    plt.show()
