"""
Build multiple charts from Soap_Agents/Output/*/agent3_result.json and meta.json.

Run from Soap_Agents:
  uv run python scripts/visualize_output_analytics.py

Figures are written to Output/analysis/ (override with --out-dir).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style

SECTION_ORDER = ["Subjective", "Objective", "Assessment", "Plan", "Other"]

BENCHMARK_COLS = [
    "accuracy_score",
    "completeness_score",
    "communication_quality_score",
    "context_awareness_score",
    "instruction_following_score",
    "overall_score",
]

# All saved figures use the same canvas size (inches); slightly tall/wide for large fonts.
FIG_SIZE = (12.0, 7.0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalize_section(name: str) -> str:
    s = (name or "").strip()
    for canon in SECTION_ORDER:
        if s.lower() == canon.lower():
            return canon
    return s or "Other"


def load_rows(output_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for folder in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        a3 = folder / "agent3_result.json"
        if not a3.is_file():
            continue
        with open(a3, encoding="utf-8") as f:
            d = json.load(f)

        meta_path = folder / "meta.json"
        iterations: int | None = None
        if meta_path.is_file():
            with open(meta_path, encoding="utf-8") as f:
                iterations = json.load(f).get("iterations")

        cv = d.get("claim_verification") or {}
        claims = cv.get("claims") or []
        n_total = len(claims) if isinstance(claims, list) else 0
        n_unsup = sum(
            1
            for c in claims
            if isinstance(c, dict) and c.get("support_status") == "unsupported"
        )

        row: dict = {
            "item": folder.name,
            "iterations": iterations,
            "n_claims": n_total,
            "n_unsupported": n_unsup,
            "unsupported_rate": (n_unsup / n_total) if n_total else np.nan,
        }
        for col in BENCHMARK_COLS:
            v = d.get(col)
            row[col] = float(v) if v is not None and v != "" else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_support_by_section(output_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (supported_counts, unsupported_counts, section_labels) aligned to SECTION_ORDER + extras."""
    sup: dict[str, int] = defaultdict(int)
    uns: dict[str, int] = defaultdict(int)

    for folder in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        a3 = folder / "agent3_result.json"
        if not a3.is_file():
            continue
        with open(a3, encoding="utf-8") as f:
            d = json.load(f)
        claims = (d.get("claim_verification") or {}).get("claims") or []
        if not isinstance(claims, list):
            continue
        for cl in claims:
            if not isinstance(cl, dict):
                continue
            sec = _normalize_section(str(cl.get("soap_section", "")))
            st = cl.get("support_status")
            if st == "supported":
                sup[sec] += 1
            elif st == "unsupported":
                uns[sec] += 1

    sections = [s for s in SECTION_ORDER if sup[s] + uns[s] > 0]
    extras = sorted({*sup, *uns} - set(SECTION_ORDER))
    sections = sections + extras
    return (
        np.array([sup[s] for s in sections], dtype=float),
        np.array([uns[s] for s in sections], dtype=float),
        sections,
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_overall_histogram(df: pd.DataFrame, out: Path) -> None:
    s = df["overall_score"].dropna()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.hist(s, bins=20, color="#2b6cb0", edgecolor="white", linewidth=0.8)
    ax.axvline(s.mean(), color="#c53030", linestyle="--", linewidth=2.2, label=f"mean = {s.mean():.3f}")
    ax.set_xlabel("Agent 2 overall score")
    ax.set_ylabel("Number of items")
    ax.set_title("Distribution of overall SOAP evaluation score")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)


def chart_boxplot_dimensions(df: pd.DataFrame, out: Path) -> None:
    plot_cols = [c for c in BENCHMARK_COLS if c != "overall_score"]
    data = [df[c].dropna().values for c in plot_cols]
    labels = [
        "Accuracy",
        "Completeness",
        "Communication",
        "Context",
        "Instruction",
    ]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#bee3f8")
        patch.set_edgecolor("#2c5282")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Agent 2 benchmark sub-scores (per item)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, out)


def chart_scatter_overall_vs_unsupported(df: pd.DataFrame, out: Path) -> None:
    sub = df.dropna(subset=["overall_score", "unsupported_rate"])
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if sub.empty:
        ax.text(
            0.5,
            0.5,
            "No rows with both scores",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=22,
        )
        ax.set_axis_off()
        fig.tight_layout()
        _save(fig, out)
        return
    sc = ax.scatter(
        sub["overall_score"],
        sub["unsupported_rate"],
        c=sub["n_claims"],
        cmap="viridis",
        alpha=0.65,
        edgecolors="white",
        linewidth=0.5,
        s=110,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Total claims (Agent 3)")
    cb.ax.tick_params(labelsize=16)
    ax.set_xlabel("Agent 2 overall score")
    ax.set_ylabel("Unsupported claim fraction (Agent 3)")
    ax.set_title("Overall benchmark vs transcript grounding")
    ax.set_ylim(-0.02, min(1.05, sub["unsupported_rate"].max() * 1.15 + 0.05))
    fig.tight_layout()
    _save(fig, out)


def chart_iterations(df: pd.DataFrame, out: Path) -> None:
    s = df["iterations"].dropna()
    counts = s.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(counts.index.astype(int).astype(str), counts.values, color="#38a169", edgecolor="#276749")
    ax.set_xlabel("Pipeline iterations (meta.json)")
    ax.set_ylabel("Number of items")
    ax.set_title("How many optimization loops ran before save")
    fig.tight_layout()
    _save(fig, out)


def chart_stacked_support_by_section(
    supported: np.ndarray,
    unsupported: np.ndarray,
    labels: list[str],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x = np.arange(len(labels))
    ax.bar(x, supported, label="Supported", color="#38a169", edgecolor="#276749", linewidth=0.5)
    ax.bar(x, unsupported, bottom=supported, label="Unsupported", color="#c53030", edgecolor="#742a2a", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Claim count (all items, summed)")
    ax.set_title("Agent 3 claims by SOAP section and support status")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out)


def chart_correlation_heatmap(df: pd.DataFrame, out: Path) -> None:
    cols = [
        "overall_score",
        "accuracy_score",
        "completeness_score",
        "communication_quality_score",
        "context_awareness_score",
        "instruction_following_score",
        "unsupported_rate",
        "n_claims",
    ]
    sub = df[[c for c in cols if c in df.columns]]
    corr = sub.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    short = {
        "communication_quality_score": "comm.",
        "instruction_following_score": "instr.",
        "context_awareness_score": "context",
        "unsupported_rate": "unsup. rate",
        "n_claims": "n claims",
    }
    lbl = [short.get(c, c.replace("_score", "").replace("_", "\n")) for c in corr.columns]
    ax.set_xticklabels(lbl, rotation=45, ha="right")
    ax.set_yticklabels(lbl)
    ax.set_title("Correlation: Agent 2 scores, claim counts, unsupported rate")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    cell_fs = int(plt.rcParams["font.size"] * 0.9)
    # annotate
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            if np.isnan(v):
                t = ""
            else:
                t = f"{v:.2f}" if abs(v) < 0.995 else f"{v:.1f}"
            ax.text(
                j,
                i,
                t,
                ha="center",
                va="center",
                color="black" if abs(v) < 0.5 else "white",
                fontsize=cell_fs,
            )
    fig.tight_layout()
    _save(fig, out)


def chart_mean_benchmark_bars(df: pd.DataFrame, out: Path) -> None:
    plot_cols = [c for c in BENCHMARK_COLS if c != "overall_score"]
    means = [df[c].mean(skipna=True) for c in plot_cols]
    labels = ["Accuracy", "Completeness", "Communication", "Context", "Instruction"]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(means)))
    bars = ax.bar(labels, means, color=colors, edgecolor="#1a365d", linewidth=0.6)
    ax.axhline(df["overall_score"].mean(skipna=True), color="#c53030", linestyle="--", label="overall mean")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score")
    ax.set_title("Mean Agent 2 dimension scores across all items")
    for b, m in zip(bars, means, strict=True):
        if not np.isnan(m):
            ax.text(
                b.get_x() + b.get_width() / 2,
                m,
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=int(plt.rcParams["font.size"]),
            )
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, out)


def chart_unsupported_per_item_top(df: pd.DataFrame, out: Path, top_n: int = 25) -> None:
    sub = df.nlargest(top_n, "n_unsupported")[["item", "n_unsupported", "n_claims", "overall_score"]]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    y = np.arange(len(sub))
    ax.barh(y, sub["n_unsupported"].values, color="#c53030", height=0.7, edgecolor="#742a2a")
    ax.set_yticks(y)
    label_fs = max(14, min(22, int(420 / max(len(sub), 1))))
    ax.set_yticklabels(sub["item"].values, fontsize=label_fs)
    ax.invert_yaxis()
    ax.set_xlabel("Unsupported claims")
    ax.set_title(f"Top {top_n} items by unsupported claim count")
    fig.tight_layout()
    _save(fig, out)


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Visualization bundle for Soap_Agents/Output.")
    ap.add_argument("--output-dir", type=Path, default=root / "Output", help="Output folder with item_* dirs.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "Output" / "analysis",
        help="Directory to write PNG files into.",
    )
    ap.add_argument("--top-n", type=int, default=25, help="Items in horizontal bar chart.")
    args = ap.parse_args()

    plot_style.apply_large_fonts()

    output_dir = args.output_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not output_dir.is_dir():
        print(f"Not found: {output_dir}", file=sys.stderr)
        return 1

    df = load_rows(output_dir)
    if df.empty:
        print("No agent3_result.json rows found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} items from {output_dir}")

    chart_overall_histogram(df, out_dir / "viz_overall_score_hist.png")
    chart_boxplot_dimensions(df, out_dir / "viz_benchmark_boxplot.png")
    chart_scatter_overall_vs_unsupported(df, out_dir / "viz_overall_vs_unsupported.png")
    chart_iterations(df, out_dir / "viz_pipeline_iterations.png")
    sup, uns, labels = aggregate_support_by_section(output_dir)
    chart_stacked_support_by_section(sup, uns, labels, out_dir / "viz_claims_stacked_by_section.png")
    chart_correlation_heatmap(df, out_dir / "viz_correlation_heatmap.png")
    chart_mean_benchmark_bars(df, out_dir / "viz_mean_benchmark_bars.png")
    chart_unsupported_per_item_top(df, out_dir / "viz_top_unsupported_items.png", top_n=args.top_n)

    # Optional CSV for notebooks
    csv_path = out_dir / "summary_per_item.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Wrote figures to {out_dir}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
