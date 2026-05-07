from __future__ import annotations

import argparse
from pathlib import Path

from .runner import RunConfig, run_batch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agent GT Validator: compare generated SOAP vs ground truth.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Output",
        help="Path containing item_* folders (default: Soap_Agents/Output).",
    )
    p.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Output" / "analysis",
        help="Where to write gt_validator_summary.csv (default: Output/analysis).",
    )
    p.add_argument("--limit", type=int, default=None, help="Only process first N item folders.")
    p.add_argument("--skip-llm", action="store_true", help="Skip Gemini judge step.")
    p.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override Gemini judge model (default env GTJUDGE_GEMINI_MODEL or gemini-2.5-flash).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = RunConfig(
        output_dir=args.output_dir.resolve(),
        analysis_dir=args.analysis_dir.resolve(),
        limit=args.limit,
        skip_llm=bool(args.skip_llm),
        judge_model=args.judge_model,
    )
    n = run_batch(cfg)
    print(f"Wrote gt_validator.json for {n} items")
    print(f"Wrote: {cfg.analysis_dir / 'gt_validator_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

