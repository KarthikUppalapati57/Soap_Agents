"""
Aggregate Agent 3 claim verification across Soap_Agents/Output/*/agent3_result.json
and plot unsupported claim counts by SOAP section.

Run from repo root:
  cd Soap_Agents && uv sync && uv run python scripts/plot_agent3_unsupported_by_section.py

Or as a module (if package layout is adjusted):
  uv run python scripts/plot_agent3_unsupported_by_section.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import plot_style

SECTION_ORDER = ["Subjective", "Objective", "Assessment", "Plan", "Other"]

# Match visualize_output_analytics.py so all pipeline figures share one footprint.
FIG_SIZE = (12.0, 7.0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalize_section(name: str) -> str:
    s = (name or "").strip()
    for canon in SECTION_ORDER:
        if s.lower() == canon.lower():
            return canon
    return s or "Other"


def load_unsupported_by_section(agent3_path: Path) -> tuple[Counter[str], str | None]:
    """
    Returns (counter of unsupported claims by soap_section, error message if skipped).
    """
    try:
        with open(agent3_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return Counter(), f"read/json error: {e}"

    cv = data.get("claim_verification")
    if cv is None:
        err = data.get("claim_verification_error")
        return Counter(), f"no claim_verification ({err or 'unknown'})"

    claims = cv.get("claims")
    if not isinstance(claims, list):
        return Counter(), "claim_verification.claims missing or not a list"

    c: Counter[str] = Counter()
    for cl in claims:
        if not isinstance(cl, dict):
            continue
        if cl.get("support_status") != "unsupported":
            continue
        sec = _normalize_section(str(cl.get("soap_section", "")))
        c[sec] += 1
    return c, None


def collect_counts(output_dir: Path) -> tuple[Counter[str], list[tuple[str, str]], int]:
    """Aggregate across child folders; return (totals, skips, n_folders_with_claim_verification)."""
    totals: Counter[str] = Counter()
    skipped: list[tuple[str, str]] = []
    n_used = 0

    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    subdirs = sorted(p for p in output_dir.iterdir() if p.is_dir())
    for folder in subdirs:
        agent3 = folder / "agent3_result.json"
        if not agent3.is_file():
            skipped.append((folder.name, "missing agent3_result.json"))
            continue
        partial, err = load_unsupported_by_section(agent3)
        if err and not partial:
            skipped.append((folder.name, err))
            continue
        n_used += 1
        if err:
            skipped.append((folder.name, err))
        totals.update(partial)

    return totals, skipped, n_used


def ordered_sections(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["Subjective", "Objective", "Assessment", "Plan"]
    seen = set(counter)
    out = [s for s in SECTION_ORDER if s in seen]
    rest = sorted(s for s in seen if s not in SECTION_ORDER)
    return out + rest


def plot_bar(counter: Counter[str], out_path: Path, title: str, *, show: bool) -> None:
    import matplotlib.pyplot as plt

    plot_style.apply_large_fonts()

    sections = ordered_sections(counter)
    values = [counter[s] for s in sections]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.bar(sections, values, color="#2c5282", edgecolor="#1a365d", linewidth=0.8)
    ax.set_ylabel("Unsupported claims (count)")
    ax.set_xlabel("SOAP section")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    for b, v in zip(bars, values, strict=True):
        if v > 0:
            ax.text(
                b.get_x() + b.get_width() / 2,
                v,
                str(int(v)),
                ha="center",
                va="bottom",
                fontsize=int(plt.rcParams["font.size"]),
            )
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_per_folder_csv(
    output_dir: Path,
    csv_path: Path,
) -> None:
    """One row per item folder: unsupported counts per section."""
    rows: list[dict[str, int | str]] = []
    subdirs = sorted(p for p in output_dir.iterdir() if p.is_dir())
    for folder in subdirs:
        agent3 = folder / "agent3_result.json"
        if not agent3.is_file():
            continue
        c, _ = load_unsupported_by_section(agent3)
        row: dict[str, int | str] = {"folder": folder.name}
        for s in SECTION_ORDER:
            row[s] = c.get(s, 0)
        for k, v in c.items():
            if k not in SECTION_ORDER:
                row[k] = v
        rows.append(row)

    if not rows:
        return

    import csv

    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = ["folder"] + [k for k in SECTION_ORDER if k in all_keys]
    fieldnames += sorted(k for k in all_keys if k not in {"folder", *SECTION_ORDER})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    root = _repo_root()
    p = argparse.ArgumentParser(
        description="Plot unsupported Agent 3 claims by SOAP section across Output folders."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Output",
        help="Directory containing item_* folders (default: Soap_Agents/Output).",
    )
    p.add_argument(
        "--figure",
        type=Path,
        default=root / "Output" / "analysis" / "unsupported_claims_by_soap_section.png",
        help="Path to write the bar chart PNG.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write per-folder unsupported counts (CSV).",
    )
    p.add_argument("--show", action="store_true", help="Display the figure interactively.")
    args = p.parse_args()

    output_dir = args.output_dir.resolve()
    try:
        totals, skipped, n_used = collect_counts(output_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    n_folders = len([p for p in output_dir.iterdir() if p.is_dir()])
    print(f"Output dir: {output_dir}")
    print(
        f"Subfolders: {n_folders}, agent3_result.json used for aggregation: {n_used}, "
        f"notes/skips logged: {len(skipped)}"
    )
    if skipped and len(skipped) <= 20:
        for name, reason in skipped:
            print(f"  skip {name}: {reason}")
    elif skipped:
        print(f"  (first 10 skips)")
        for name, reason in skipped[:10]:
            print(f"  skip {name}: {reason}")

    total_unsupported = sum(totals.values())
    print(f"Total unsupported claims (all sections): {total_unsupported}")
    for sec in ordered_sections(totals):
        print(f"  {sec}: {totals[sec]}")

    if n_used == 0:
        print("No agent3_result.json with claim_verification found; nothing to plot.", file=sys.stderr)
        return 1

    plot_bar(
        totals,
        args.figure.resolve(),
        "Unsupported claims by SOAP section (all Output items)",
        show=args.show,
    )
    print(f"Wrote figure: {args.figure.resolve()}")

    if args.csv:
        write_per_folder_csv(output_dir, args.csv.resolve())
        print(f"Wrote CSV: {args.csv.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
