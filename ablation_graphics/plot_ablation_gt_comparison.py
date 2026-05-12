"""
Side-by-side ablation comparison for three GT validator summary CSVs.

Reads gt_validator_summary.csv from each run (default: Output_baseline, Output_no_mkg,
Output_full) and writes three separate bar-chart PNGs:

1. Mean letter grade (A=4, B=3, C=2, D=1, F=0; rows without a valid grade excluded).
2. Mean count of judge additions unsupported by both transcript and PrimeKG.
3. Mean count of judge additions supported by transcript or PrimeKG.

Run from Soap_Agents/:
  uv run python "ablation graphics/plot_ablation_gt_comparison.py"

Figures and a summary CSV are written to this directory (ablation graphics/) by default.
The CSV has one row per output with the bar-chart metrics and supporting counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

_SOAP_AGENTS_ROOT = Path(__file__).resolve().parent.parent
if str(_SOAP_AGENTS_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_SOAP_AGENTS_ROOT / "scripts"))
import plot_style  # noqa: E402


GRADE_TO_NUM = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}


def _letter_grade_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    return s.map(GRADE_TO_NUM)


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    return pd.read_csv(path)


def _bar_single_figure(
    *,
    values: list[float],
    bar_labels: list[str],
    title: str,
    ylabel: str,
    out_path: Path,
    y_top_cap: float | None = None,
    figsize: tuple[float, float] = (7.2, 5.0),
) -> None:
    x = np.arange(len(bar_labels))
    width = 0.55
    colors = ("#3182ce", "#38a169", "#805ad5")

    clean = [float(v) if np.isfinite(v) else 0.0 for v in values]
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.bar(x, clean, width, color=colors, edgecolor="#1a202c", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ymax = max(clean) if clean else 0.0
    pad = max(0.06 * ymax, 0.06)
    top = ymax + pad
    if y_top_cap is not None:
        top = min(y_top_cap, top)
    ax.set_ylim(0.0, top)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    default_root = _SOAP_AGENTS_ROOT
    default_out_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description="Ablation bar charts for GT validator metrics.")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=default_root / "Output_baseline" / "analysis" / "gt_validator_summary.csv",
        help="Path to baseline gt_validator_summary.csv",
    )
    ap.add_argument(
        "--no-mkg",
        type=Path,
        dest="no_mkg",
        default=default_root / "Output_no_mkg" / "analysis" / "gt_validator_summary.csv",
        help="Path to no-MKG gt_validator_summary.csv",
    )
    ap.add_argument(
        "--full",
        type=Path,
        default=default_root / "Output_full" / "analysis" / "gt_validator_summary.csv",
        help="Path to full gt_validator_summary.csv",
    )
    ap.add_argument(
        "--labels",
        nargs=3,
        default=["Base", "No KG", "Full"],
        metavar=("L0", "L1", "L2"),
        help="Bar labels for the three conditions (left to right).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Directory for output PNGs and CSV.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Summary CSV path (default: <out-dir>/ablation_gt_comparison_values.csv).",
    )
    args = ap.parse_args()

    paths = [args.baseline.resolve(), args.no_mkg.resolve(), args.full.resolve()]
    bar_labels = list(args.labels)
    names = ["baseline", "no_mkg", "full"]

    plot_style.apply_large_fonts()

    frames = {n: _load_summary(p) for n, p in zip(names, paths, strict=True)}

    col_both = "n_additions_unsupported_by_transcript_and_primekg"
    col_sup = "n_additions_supported_by_transcript_or_primekg"

    rows: list[dict[str, object]] = []
    for i, n in enumerate(names):
        df = frames[n]
        g = _letter_grade_numeric(df["judge_overall_grade"])
        valid = g.notna()
        both = pd.to_numeric(df[col_both], errors="coerce")
        sup = pd.to_numeric(df[col_sup], errors="coerce")
        rows.append(
            {
                "output": n,
                "label": bar_labels[i],
                "summary_csv": str(paths[i]),
                "n_items": len(df),
                "mean_letter_grade": float(g[valid].mean()) if valid.any() else float("nan"),
                "n_items_letter_grade": int(valid.sum()),
                "mean_additions_unsupported_transcript_and_primekg": float(both.mean(skipna=True)),
                "n_items_additions_unsupported_both": int(both.notna().sum()),
                "mean_additions_supported_transcript_or_primekg": float(sup.mean(skipna=True)),
                "n_items_additions_supported_tr_or_pk": int(sup.notna().sum()),
            }
        )

    out_df = pd.DataFrame(rows)
    mean_grade = out_df["mean_letter_grade"].astype(float).tolist()
    mean_both_unsup = out_df["mean_additions_unsupported_transcript_and_primekg"].astype(float).tolist()
    mean_tr_or_pk_sup = out_df["mean_additions_supported_transcript_or_primekg"].astype(float).tolist()

    out_dir = args.out_dir.resolve()
    csv_path = (args.out_csv or (out_dir / "ablation_gt_comparison_values.csv")).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    outputs = [
        (
            out_dir / "ablation_mean_letter_grade.png",
            mean_grade,
            "Mean Judge Grade",
            "Grade (A=4 to F=0)",
            4.35,
        ),
        (
            out_dir / "ablation_additions_unsupported_transcript_and_primekg.png",
            mean_both_unsup,
            "Unsupported Additions",
            "Mean count per item",
            None,
        ),
        (
            out_dir / "ablation_additions_supported_transcript_or_primekg.png",
            mean_tr_or_pk_sup,
            "Supported Additions",
            "Mean count per item",
            None,
        ),
    ]

    for path, vals, title, ylabel, cap in outputs:
        _bar_single_figure(
            values=vals,
            bar_labels=bar_labels,
            title=title,
            ylabel=ylabel,
            out_path=path,
            y_top_cap=cap,
        )
        print(f"Wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
