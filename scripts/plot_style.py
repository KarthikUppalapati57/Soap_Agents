"""Shared matplotlib typography for Soap_Agents report figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_large_fonts() -> None:
    """Bump default text sizes substantially (titles, axis labels, ticks, legend)."""
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 26,
            "axes.titleweight": "bold",
            "axes.labelsize": 22,
            "axes.labelweight": "bold",
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 26,
        }
    )

