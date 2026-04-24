"""
Batch claim verification over Agent-2 v2_results-style JSON.

Run from Agent-3 with uv:

  uv sync
  uv run python agent3.py --limit 2

Environment: see .env.example (GOOGLE_API_KEY or GEMINI_API_KEY / gemini_api_key).
Optional: AGENT3_GEMINI_MODEL (default gemini-2.5-flash).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from env_setup import load_env, require_api_key
from runner import benchmark_from_row, verify_claims_async
from schemas import BenchmarkScores, ClaimVerificationInput


def _default_input_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "Agent-2"
        / "v2"
        / "OpenAI health Benchmark"
        / "v2_results.json"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agent 3: SOAP claim verification (Google ADK).")
    p.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Path to v2_results.json (or compatible list of rows).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "v2_with_claims.json",
        help="Enriched JSON output path.",
    )
    p.add_argument("--limit", type=int, default=None, help="Max new rows to process.")
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max concurrent claim-verification requests (default: 4).",
    )
    p.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated row ids to include (default: all pending).",
    )
    return p.parse_args()


async def _process_one(row: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        rid = row.get("id")
        payload = ClaimVerificationInput(
            transcript=row["transcript"],
            generated_soap=row["generated"],
            benchmark=BenchmarkScores.model_validate(benchmark_from_row(row)),
        )
        try:
            out = await verify_claims_async(payload)
            return {
                **row,
                "claim_verification": out.model_dump(),
                "claim_verification_error": None,
            }
        except Exception as e:
            print(f"  Error id={rid}: {e}", file=sys.stderr)
            return {
                **row,
                "claim_verification": None,
                "claim_verification_error": str(e),
            }


async def _run_batch(args: argparse.Namespace) -> None:
    load_env()
    require_api_key()

    input_path: Path = args.input
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_name(output_path.stem + "_checkpoint.json")

    with open(input_path, encoding="utf-8") as f:
        source_rows: list[dict] = json.load(f)

    id_filter: set[int] | None = None
    if args.ids:
        id_filter = {int(x.strip()) for x in args.ids.split(",") if x.strip()}

    results: list[dict] = []
    done_ids: set[int] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {int(r["id"]) for r in results}

    rows_to_process: list[dict] = []
    for row in source_rows:
        rid = row.get("id")
        if rid is None:
            continue
        if id_filter is not None and int(rid) not in id_filter:
            continue
        if int(rid) in done_ids:
            continue
        if row.get("status") != "OK":
            continue

        rows_to_process.append(row)
        if args.limit is not None and len(rows_to_process) >= args.limit:
            break

    if not rows_to_process:
        print("No pending rows to process.")
        return

    workers = max(1, int(args.workers))
    sem = asyncio.Semaphore(workers)

    print(f"Processing {len(rows_to_process)} rows with workers={workers} ...")

    tasks = [asyncio.create_task(_process_one(row, sem)) for row in rows_to_process]
    processed = 0
    for fut in asyncio.as_completed(tasks):
        row_out = await fut
        rid = row_out.get("id")
        print(f"Done id={rid}")

        results.append(row_out)
        if rid is not None:
            done_ids.add(int(rid))
        processed += 1

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    exit_due_to_limit = args.limit is not None and processed >= args.limit
    if not exit_due_to_limit and checkpoint_path.exists():
        checkpoint_path.unlink(missing_ok=True)

    print(f"Wrote {len(results)} rows to {output_path}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_run_batch(args))


if __name__ == "__main__":
    main()
