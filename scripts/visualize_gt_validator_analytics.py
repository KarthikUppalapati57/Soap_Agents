"""
Research-ready visualization bundle for GT validator outputs.

Inputs:
- Output/analysis/gt_validator_summary.csv (produced by Agent_GT_Validator runner)

Run from Soap_Agents/:
  uv run python scripts/visualize_gt_validator_analytics.py

Writes figures to Output/analysis/ (override with --out-dir).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing matplotlib. Run `uv sync` from the repo root to install dependencies."
    ) from e

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing numpy. Run `uv sync` from the repo root to install dependencies."
    ) from e

import plot_style


FIG_SIZE = (12.0, 7.0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _to_float(x: str | None) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _to_int(x: str | None) -> int | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def load_summary_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row for row in r]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def chart_histogram(values: np.ndarray, *, xlabel: str, title: str, out: Path) -> None:
    s = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if s.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        _save(fig, out)
        return
    ax.hist(s, bins=20, color="#2b6cb0", edgecolor="white", linewidth=0.8)
    mu = _nanmean(s)
    if np.isfinite(mu):
        ax.axvline(mu, color="#c53030", linestyle="--", linewidth=2.2, label=f"mean = {mu:.3f}")
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of items")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, out)


def chart_boxplot(metrics: list[np.ndarray], labels: list[str], *, ylabel: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    data = [m[np.isfinite(m)] for m in metrics]
    if not any(len(d) for d in data):
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        _save(fig, out)
        return
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#bee3f8")
        patch.set_edgecolor("#2c5282")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, out)


def chart_grade_counts(grades: list[str], *, title: str, out: Path) -> None:
    order = ["A", "B", "C", "D", "F"]
    counts = {g: 0 for g in order}
    for g in grades:
        gg = (g or "").strip().upper()
        if gg in counts:
            counts[gg] += 1
    xs = [g for g in order if counts[g] > 0]
    ys = [counts[g] for g in xs]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if not xs:
        ax.text(0.5, 0.5, "No judge grades found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        _save(fig, out)
        return
    ax.bar(xs, ys, color="#805ad5", edgecolor="#44337a", linewidth=0.8)
    ax.set_xlabel("Judge overall grade")
    ax.set_ylabel("Number of items")
    ax.set_title(title)
    for x, y in zip(xs, ys, strict=True):
        ax.text(x, y, str(int(y)), ha="center", va="bottom")
    fig.tight_layout()
    _save(fig, out)


def chart_scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    color: np.ndarray | None,
    cbar_label: str | None,
    out: Path,
) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if not mask.any():
        ax.text(0.5, 0.5, "No rows with both values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        _save(fig, out)
        return
    sc = ax.scatter(
        x[mask],
        y[mask],
        c=(color[mask] if color is not None else None),
        cmap="viridis" if color is not None else None,
        alpha=0.7,
        edgecolors="white",
        linewidth=0.5,
        s=110,
    )
    if color is not None and cbar_label:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save(fig, out)


def chart_correlation_heatmap(mat: np.ndarray, labels: list[str], *, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if mat.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        _save(fig, out)
        return
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], rotation=45, ha="right")
    ax.set_yticklabels([l.replace("_", "\n") for l in labels])
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    cell_fs = int(plt.rcParams["font.size"] * 0.85)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            t = "" if np.isnan(v) else f"{v:.2f}"
            ax.text(
                j,
                i,
                t,
                ha="center",
                va="center",
                color="black" if np.isnan(v) or abs(v) < 0.5 else "white",
                fontsize=cell_fs,
            )
    fig.tight_layout()
    _save(fig, out)


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Visualization bundle for GT validator outputs.")
    ap.add_argument(
        "--summary-csv",
        type=Path,
        default=root / "Output" / "analysis" / "gt_validator_summary.csv",
        help="Path to gt_validator_summary.csv.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "Output" / "analysis" / "gt_validator",
        help="Directory to write PNG figures into.",
    )
    args = ap.parse_args()

    plot_style.apply_large_fonts()

    rows = load_summary_csv(args.summary_csv.resolve())
    if not rows:
        print("Summary CSV had no rows.", file=sys.stderr)
        return 1

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # numeric columns
    recall_all = np.array([_to_float(r.get("clinical_recall_all")) for r in rows], dtype=float)
    recall_sym = np.array([_to_float(r.get("clinical_recall_symptoms")) for r in rows], dtype=float)
    recall_med = np.array([_to_float(r.get("clinical_recall_medications")) for r in rows], dtype=float)
    recall_lab = np.array([_to_float(r.get("clinical_recall_labs")) for r in rows], dtype=float)

    tok_f1 = np.array([_to_float(r.get("overall_token_f1")) for r in rows], dtype=float)
    rouge = np.array([_to_float(r.get("overall_rouge_l_f1")) for r in rows], dtype=float)
    assess = np.array([_to_float(r.get("assessment_alignment_score")) for r in rows], dtype=float)
    plan = np.array([_to_float(r.get("plan_alignment_score")) for r in rows], dtype=float)

    add_sup = np.array([_to_float(r.get("n_additions_supported_by_transcript")) for r in rows], dtype=float)
    add_unsup = np.array([_to_float(r.get("n_additions_unsupported_by_transcript")) for r in rows], dtype=float)
    add_unk = np.array([_to_float(r.get("n_additions_unknown_by_transcript")) for r in rows], dtype=float)
    omit_unsup = np.array([_to_float(r.get("n_omissions_unsupported_by_transcript")) for r in rows], dtype=float)

    grades = [str(r.get("judge_overall_grade") or "").strip() for r in rows]

    # 1) Distributions
    chart_histogram(
        recall_all,
        xlabel="Clinical recall (all entities) vs GT",
        title="Distribution of entity recall (GT vs generated)",
        out=out_dir / "viz_gt_entity_recall_all_hist.png",
    )
    chart_histogram(
        assess,
        xlabel="Assessment alignment (token-F1)",
        title="Distribution of Assessment section alignment (GT vs generated)",
        out=out_dir / "viz_gt_assessment_alignment_hist.png",
    )

    # 2) Boxplot of recalls
    chart_boxplot(
        [recall_all, recall_sym, recall_med, recall_lab],
        ["All", "Symptoms", "Meds", "Labs"],
        ylabel="Recall",
        title="Clinical recall by category (GT vs generated)",
        out=out_dir / "viz_gt_recall_boxplot.png",
    )

    # 3) Alignment metrics boxplot
    chart_boxplot(
        [tok_f1, rouge, assess, plan],
        ["overall_token_F1", "overall_ROUGE-L", "assessment_token_F1", "plan_token_F1"],
        ylabel="Score",
        title="Structural alignment (GT vs generated)",
        out=out_dir / "viz_gt_alignment_boxplot.png",
    )

    # 4) Judge grade histogram
    chart_grade_counts(
        grades,
        title="GT judge overall grades (distribution)",
        out=out_dir / "viz_gt_judge_grade_counts.png",
    )

    # 4b) Transcript unsupported counts (judge-based)
    chart_histogram(
        add_unsup,
        xlabel="Additions unsupported by transcript (count)",
        title="Distribution of judge additions unsupported by transcript",
        out=out_dir / "viz_gt_additions_unsupported_by_transcript_hist.png",
    )
    chart_histogram(
        omit_unsup,
        xlabel="Omissions unsupported by transcript (count)",
        title="Distribution of judge omissions unsupported by transcript (GT-only)",
        out=out_dir / "viz_gt_omissions_unsupported_by_transcript_hist.png",
    )

    # 5) Scatter: recall vs alignment (colored by supported additions)
    chart_scatter(
        recall_all,
        tok_f1,
        xlabel="Entity recall (all)",
        ylabel="Overall alignment (token-F1)",
        title="Recall vs alignment (color = supported additions count)",
        color=(add_sup if np.isfinite(add_sup).any() else None),
        cbar_label="Additions supported by transcript",
        out=out_dir / "viz_gt_recall_vs_alignment_scatter.png",
    )

    # 6) Correlation heatmap
    cols = {
        "recall_all": recall_all,
        "recall_symptoms": recall_sym,
        "recall_meds": recall_med,
        "recall_labs": recall_lab,
        "overall_token_f1": tok_f1,
        "assessment_token_f1": assess,
        "plan_token_f1": plan,
        "additions_supported": add_sup,
        "additions_unsupported": add_unsup,
        "omissions_unsupported": omit_unsup,
    }
    labels = list(cols.keys())
    X = np.vstack([cols[k] for k in labels]).T
    # pairwise corr with NaN handling
    corr = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for i in range(len(labels)):
        for j in range(len(labels)):
            xi = X[:, i]
            xj = X[:, j]
            m = np.isfinite(xi) & np.isfinite(xj)
            if m.sum() < 3:
                continue
            corr[i, j] = float(np.corrcoef(xi[m], xj[m])[0, 1])
    chart_correlation_heatmap(
        corr,
        labels,
        title="Correlation: GT recall/alignment vs transcript-supported additions",
        out=out_dir / "viz_gt_correlation_heatmap.png",
    )

    print(f"Loaded {len(rows)} rows from {args.summary_csv.resolve()}")
    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

