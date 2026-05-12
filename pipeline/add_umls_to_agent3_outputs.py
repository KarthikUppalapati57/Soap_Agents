#!/usr/bin/env python3
"""
Walk an output directory (e.g. Output_full/) and enrich each item's agent3 JSON with
UMLS term validation, matching ``AgentInterface.run_agent_2`` in ``agent_interface.py``:

- ``UMLS_valid_terms``, ``UMLS_invalid_terms`` from GLiNER entities + UMLS search API
- ``UMLS_accuracy_score`` = len(valid) / (len(valid) + len(invalid)), or 0.0 if no terms

Requires ``UMLS_API_KEY`` in ``.env`` (see ``Agent_2/v2/MKG/mkg_validation.py``), ``gliner``
(see ``Agent_3/MKG.py``), and network for UMLS / first-time model download.

Usage (from Soap_Agents repo root)::

    uv run python pipeline/add_umls_to_agent3_outputs.py Output_full

    uv run python pipeline/add_umls_to_agent3_outputs.py Output_full --result-file agent3_results.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_import_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _iter_item_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {output_dir}")
    subdirs = [p for p in output_dir.iterdir() if p.is_dir()]
    prefix_re = re.compile(r"^item_(\d+)$")

    def sort_key(p: Path) -> tuple[int, str]:
        m = prefix_re.match(p.name)
        if m:
            return (int(m.group(1)), p.name)
        return (10**9, p.name)

    return sorted(subdirs, key=sort_key)


def _umls_accuracy(valid: list[str], invalid: list[str]) -> float:
    n = len(valid) + len(invalid)
    if n == 0:
        return 0.0
    return len(valid) / n


def _process_file(
    path: Path,
    *,
    extract_terms: Callable[[str], list[str]],
    validate_terms: Callable[[list[str]], tuple[list[str], list[str]]],
    dry_run: bool,
) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    generated = data.get("generated")
    if not isinstance(generated, str) or not generated.strip():
        return {
            "path": str(path),
            "status": "skipped",
            "reason": "missing_or_empty_generated",
        }

    terms = extract_terms(generated)
    valid_terms, invalid_terms = validate_terms(terms)
    data["UMLS_valid_terms"] = valid_terms
    data["UMLS_invalid_terms"] = invalid_terms
    data["UMLS_accuracy_score"] = _umls_accuracy(valid_terms, invalid_terms)

    if not dry_run:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    return {
        "path": str(path),
        "status": "ok",
        "term_count": len(terms),
        "valid_count": len(valid_terms),
        "invalid_count": len(invalid_terms),
        "UMLS_accuracy_score": data["UMLS_accuracy_score"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Add UMLS validation fields to per-item agent3 JSON files.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory containing item_* subfolders (e.g. Output_full).",
    )
    parser.add_argument(
        "--result-file",
        default="agent3_result.json",
        help="JSON filename inside each item folder (default: agent3_result.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute metrics but do not write files.",
    )
    args = parser.parse_args()

    _ensure_import_path()
    os.chdir(_repo_root())

    try:
        from dotenv import load_dotenv

        load_dotenv(_repo_root() / ".env")
    except ImportError:
        pass

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (_repo_root() / output_dir).resolve()

    result_name = args.result_file
    from Agent_2.v2.MKG.mkg_validation import extract_terms, validate_terms

    summaries: list[dict] = []
    for item_dir in _iter_item_dirs(output_dir):
        json_path = item_dir / result_name
        if not json_path.is_file():
            summaries.append(
                {
                    "path": str(json_path),
                    "status": "skipped",
                    "reason": f"missing_file:{result_name}",
                }
            )
            continue
        try:
            summaries.append(
                _process_file(
                    json_path,
                    extract_terms=extract_terms,
                    validate_terms=validate_terms,
                    dry_run=args.dry_run,
                )
            )
        except Exception as e:
            summaries.append(
                {
                    "path": str(json_path),
                    "status": "error",
                    "error": str(e),
                }
            )

    ok = sum(1 for s in summaries if s.get("status") == "ok")
    skipped = sum(1 for s in summaries if s.get("status") == "skipped")
    errors = sum(1 for s in summaries if s.get("status") == "error")
    scores = [s["UMLS_accuracy_score"] for s in summaries if s.get("status") == "ok" and "UMLS_accuracy_score" in s]
    mean_score = sum(scores) / len(scores) if scores else None

    print(f"Output directory: {output_dir}")
    print(f"Result file: {result_name}")
    print(f"Processed ok: {ok} | skipped: {skipped} | errors: {errors}")
    if mean_score is not None:
        print(f"Mean UMLS_accuracy_score (ok items): {mean_score:.4f}")
    if args.dry_run:
        print("(dry-run: no files written)")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
