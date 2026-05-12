"""Compute ROUGE-1, ROUGE-2, and ROUGE-L scores for SOAP-note results.

Two input modes are supported:

1. **File mode** — a JSON file produced by ``V1/run.py`` (a list of records
   with ``generated`` and ``ground_truth`` / ``reference_text`` fields). A
   single summary JSON is emitted with per-sample scores and a corpus-level
   average.

2. **Directory mode** — a pipeline output directory such as
   ``Soap_Agents/Output_full/`` that contains ``item_*`` subfolders with
   ``source.json`` (``ground_truth``) and ``agent3_result.json``
   (``generated``). A ``rouge.json`` file is written into each item folder
   and a ``rouge_summary.json`` is written at the directory root.

Usage (CLI):
    # File mode
    python evaluate_rouge.py path/to/results.json \
        --output results/rouge_eval.json \
        --prompt-type one_shot

    # Directory mode (writes rouge.json into every item_*/ folder)
    python evaluate_rouge.py /path/to/Output_full

Usage (Python):
    from Agent_1.evaluate_rouge import evaluate_rouge_file, evaluate_rouge_directory
    summary = evaluate_rouge_file("results/v1_results.json")
    summary = evaluate_rouge_directory("Output_full")
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Iterable

from rouge_score import rouge_scorer

ROUGE_TYPES: tuple[str, ...] = ("rouge1", "rouge2", "rougeL")

DEFAULT_ITEM_GLOB_RE = re.compile(r"^item_\d+$")
DEFAULT_SOURCE_FILE = "source.json"
DEFAULT_PREDICTION_FILE = "agent3_result.json"
DEFAULT_META_FILE = "meta.json"
DEFAULT_ROUGE_FILENAME = "rouge.json"
DEFAULT_SUMMARY_FILENAME = "rouge_summary.json"


def _extract_pair(record: dict[str, Any]) -> tuple[str, str] | None:
    """Return ``(prediction, reference)`` from a result record, or ``None``.

    Accepts both schemas seen in the project:
      - ``v1_results.json`` uses ``generated`` and ``ground_truth``
      - ``V1/run.py`` writes ``generated`` and ``reference_text``
    """
    prediction = record.get("generated")
    reference = record.get("ground_truth") or record.get("reference_text")
    if not prediction or not reference:
        return None
    return str(prediction).strip(), str(reference).strip()


def _round(value: float, ndigits: int) -> float:
    return round(float(value), ndigits)


def _make_scorer() -> rouge_scorer.RougeScorer:
    return rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)


def _score_pair(
    scorer: rouge_scorer.RougeScorer,
    prediction: str,
    reference: str,
    ndigits: int,
) -> dict[str, float]:
    scores = scorer.score(reference, prediction)
    return {rt: _round(scores[rt].fmeasure, ndigits) for rt in ROUGE_TYPES}


def compute_rouge_scores(
    records: Iterable[dict[str, Any]],
    ndigits: int = 4,
) -> dict[str, Any]:
    """Compute per-sample and average ROUGE-1/2/L F-measures.

    Args:
        records: Iterable of result dicts (must contain ``generated`` and a
            reference field).
        ndigits: Number of decimal places to keep in the output.

    Returns:
        A dict with ``average`` (corpus-level mean F-measure for each ROUGE
        variant) and ``samples`` (per-record scores).
    """
    scorer = _make_scorer()

    samples: list[dict[str, Any]] = []
    totals: dict[str, float] = {rt: 0.0 for rt in ROUGE_TYPES}
    skipped = 0

    for idx, record in enumerate(records):
        pair = _extract_pair(record)
        if pair is None:
            skipped += 1
            continue
        prediction, reference = pair

        sample_scores = _score_pair(scorer, prediction, reference, ndigits)

        sample_entry: dict[str, Any] = {"id": record.get("id", idx)}
        for rt in ROUGE_TYPES:
            sample_entry[rt] = sample_scores[rt]
            totals[rt] += sample_scores[rt]
        samples.append(sample_entry)

    n = len(samples)
    average = {rt: _round(totals[rt] / n, ndigits) if n else 0.0 for rt in ROUGE_TYPES}

    summary: dict[str, Any] = {
        "n_samples": n,
        "average": average,
        "samples": samples,
    }
    if skipped:
        summary["skipped"] = skipped
    return summary


def evaluate_rouge_file(
    input_path: str,
    output_path: str | None = None,
    prompt_type: str | None = None,
    ndigits: int = 4,
) -> dict[str, Any]:
    """Evaluate ROUGE on a results JSON file and optionally write a summary.

    Args:
        input_path: Path to a JSON file containing a list of result records.
        output_path: Optional path to write the summary JSON.
        prompt_type: Optional label (e.g. ``"zero_shot"``) included in output.
        ndigits: Decimal places to keep in the summary output.

    Returns:
        The summary dict (also written to ``output_path`` when provided).
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list of records in {input_path}, got {type(data).__name__}."
        )

    summary = compute_rouge_scores(data, ndigits=ndigits)
    if prompt_type:
        summary = {"prompt_type": prompt_type, **summary}

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def _list_item_dirs(directory: str) -> list[str]:
    """Return sorted ``item_*`` subdirectory names under ``directory``."""
    return sorted(
        name
        for name in os.listdir(directory)
        if DEFAULT_ITEM_GLOB_RE.match(name)
        and os.path.isdir(os.path.join(directory, name))
    )


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _evaluate_one_item(
    item_dir: str,
    scorer: rouge_scorer.RougeScorer,
    source_filename: str,
    prediction_filename: str,
    rouge_filename: str,
    ndigits: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Compute ROUGE for a single item folder and write ``rouge.json``.

    Returns a status dict with the item name, computed scores (when
    successful), and a ``status`` field.
    """
    item_name = os.path.basename(os.path.normpath(item_dir))
    out_path = os.path.join(item_dir, rouge_filename)
    source_path = os.path.join(item_dir, source_filename)
    prediction_path = os.path.join(item_dir, prediction_filename)

    if not overwrite and os.path.exists(out_path):
        try:
            existing = _read_json(out_path)
            existing_scores = {rt: existing.get(rt) for rt in ROUGE_TYPES}
            return {
                "item": item_name,
                "status": "skipped_existing",
                **existing_scores,
            }
        except (OSError, json.JSONDecodeError):
            pass

    if not os.path.exists(source_path) or not os.path.exists(prediction_path):
        return {
            "item": item_name,
            "status": "missing_files",
            "missing": [
                p
                for p in (source_path, prediction_path)
                if not os.path.exists(p)
            ],
        }

    try:
        source = _read_json(source_path)
        prediction_doc = _read_json(prediction_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"item": item_name, "status": "read_error", "error": str(exc)}

    reference = source.get("ground_truth") if isinstance(source, dict) else None
    prediction = (
        prediction_doc.get("generated") if isinstance(prediction_doc, dict) else None
    )

    if not reference or not prediction:
        return {
            "item": item_name,
            "status": "missing_text",
            "has_ground_truth": bool(reference),
            "has_generated": bool(prediction),
        }

    sample_scores = _score_pair(scorer, str(prediction).strip(), str(reference).strip(), ndigits)

    rouge_doc: dict[str, Any] = {"item": item_name}
    index_value: Any = None
    meta_path = os.path.join(item_dir, DEFAULT_META_FILE)
    if os.path.exists(meta_path):
        try:
            meta_doc = _read_json(meta_path)
            if isinstance(meta_doc, dict) and "index" in meta_doc:
                index_value = meta_doc["index"]
        except (OSError, json.JSONDecodeError):
            pass
    if index_value is None and isinstance(prediction_doc, dict) and "index" in prediction_doc:
        index_value = prediction_doc["index"]
    if index_value is not None:
        rouge_doc["index"] = index_value
    rouge_doc.update(sample_scores)
    rouge_doc["source_file"] = source_filename
    rouge_doc["prediction_file"] = prediction_filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rouge_doc, f, indent=2)

    return {"item": item_name, "status": "ok", **sample_scores}


def evaluate_rouge_directory(
    directory: str,
    *,
    source_filename: str = DEFAULT_SOURCE_FILE,
    prediction_filename: str = DEFAULT_PREDICTION_FILE,
    rouge_filename: str = DEFAULT_ROUGE_FILENAME,
    summary_filename: str | None = DEFAULT_SUMMARY_FILENAME,
    ndigits: int = 4,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Iterate ``item_*`` subfolders of ``directory`` and write per-item ROUGE.

    For each item folder, reads ``ground_truth`` from ``source_filename`` and
    ``generated`` from ``prediction_filename``, computes ROUGE-1/2/L
    F-measures, and writes ``rouge_filename`` inside the item folder. A
    top-level ``summary_filename`` (if provided) records corpus averages and
    per-item scores.

    Args:
        directory: Output directory (e.g. ``Output_full``).
        source_filename: File inside each item with ``ground_truth``.
        prediction_filename: File inside each item with ``generated``.
        rouge_filename: Output filename written into each item folder.
        summary_filename: Top-level summary filename. ``None`` to skip.
        ndigits: Decimal places to keep in scores.
        overwrite: If ``False``, items that already have ``rouge_filename``
            are left untouched.

    Returns:
        A summary dict with ``directory``, per-item ``items`` list, ``average``
        across successful items, and counts (``n_items``, ``n_scored``).
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"{directory} is not a directory")

    item_names = _list_item_dirs(directory)
    if not item_names:
        raise FileNotFoundError(
            f"No 'item_*' subdirectories found in {directory}"
        )

    scorer = _make_scorer()

    items: list[dict[str, Any]] = []
    totals: dict[str, float] = {rt: 0.0 for rt in ROUGE_TYPES}
    n_scored = 0

    for name in item_names:
        item_dir = os.path.join(directory, name)
        result = _evaluate_one_item(
            item_dir=item_dir,
            scorer=scorer,
            source_filename=source_filename,
            prediction_filename=prediction_filename,
            rouge_filename=rouge_filename,
            ndigits=ndigits,
            overwrite=overwrite,
        )
        items.append(result)
        if result["status"] in ("ok", "skipped_existing") and all(
            isinstance(result.get(rt), (int, float)) for rt in ROUGE_TYPES
        ):
            for rt in ROUGE_TYPES:
                totals[rt] += float(result[rt])
            n_scored += 1

    average = {
        rt: _round(totals[rt] / n_scored, ndigits) if n_scored else 0.0
        for rt in ROUGE_TYPES
    }

    summary: dict[str, Any] = {
        "directory": os.path.abspath(directory),
        "n_items": len(items),
        "n_scored": n_scored,
        "average": average,
        "items": items,
    }

    if summary_filename:
        summary_path = os.path.join(directory, summary_filename)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        summary["summary_path"] = summary_path

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute ROUGE-1/2/L scores for SOAP-note results.",
    )
    parser.add_argument(
        "input",
        help=(
            "Path to either a results JSON file (file mode) or an output "
            "directory containing item_*/ subfolders (directory mode)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "File mode: path to write the summary JSON (defaults to stdout). "
            "Directory mode: ignored (rouge.json is written into each item)."
        ),
    )
    parser.add_argument(
        "-p",
        "--prompt-type",
        default=None,
        help="File mode: optional label (e.g. 'zero_shot') stored in the output.",
    )
    parser.add_argument(
        "--ndigits",
        type=int,
        default=4,
        help="Decimal places to keep in scores (default: 4).",
    )
    parser.add_argument(
        "--source-file",
        default=DEFAULT_SOURCE_FILE,
        help=(
            "Directory mode: filename inside each item providing 'ground_truth' "
            f"(default: {DEFAULT_SOURCE_FILE})."
        ),
    )
    parser.add_argument(
        "--prediction-file",
        default=DEFAULT_PREDICTION_FILE,
        help=(
            "Directory mode: filename inside each item providing 'generated' "
            f"(default: {DEFAULT_PREDICTION_FILE})."
        ),
    )
    parser.add_argument(
        "--rouge-filename",
        default=DEFAULT_ROUGE_FILENAME,
        help=(
            "Directory mode: name of the per-item output file "
            f"(default: {DEFAULT_ROUGE_FILENAME})."
        ),
    )
    parser.add_argument(
        "--summary-filename",
        default=DEFAULT_SUMMARY_FILENAME,
        help=(
            "Directory mode: name of the directory-level summary file written "
            f"at the root (default: {DEFAULT_SUMMARY_FILENAME}). "
            "Pass an empty string to skip."
        ),
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Directory mode: skip items that already have the rouge file.",
    )
    return parser


def _run_directory_mode(args: argparse.Namespace) -> None:
    summary = evaluate_rouge_directory(
        directory=args.input,
        source_filename=args.source_file,
        prediction_filename=args.prediction_file,
        rouge_filename=args.rouge_filename,
        summary_filename=args.summary_filename or None,
        ndigits=args.ndigits,
        overwrite=not args.no_overwrite,
    )

    avg = summary["average"]
    print(
        f"Scored {summary['n_scored']}/{summary['n_items']} items in "
        f"{summary['directory']}"
    )
    print(
        f"  ROUGE-1 F1: {avg['rouge1']:.4f} | "
        f"ROUGE-2 F1: {avg['rouge2']:.4f} | "
        f"ROUGE-L F1: {avg['rougeL']:.4f}"
    )
    failures = [
        item for item in summary["items"] if item["status"] not in ("ok", "skipped_existing")
    ]
    if failures:
        print(f"  {len(failures)} item(s) had issues:")
        for item in failures[:10]:
            print(f"    - {item['item']}: {item['status']}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
    if summary.get("summary_path"):
        print(f"Wrote directory summary to {summary['summary_path']}")


def _run_file_mode(args: argparse.Namespace) -> None:
    summary = evaluate_rouge_file(
        input_path=args.input,
        output_path=args.output,
        prompt_type=args.prompt_type,
        ndigits=args.ndigits,
    )

    avg = summary["average"]
    n = summary.get("n_samples", len(summary.get("samples", [])))
    print(f"Evaluated {n} samples from {args.input}")
    print(
        f"  ROUGE-1 F1: {avg['rouge1']:.4f} | "
        f"ROUGE-2 F1: {avg['rouge2']:.4f} | "
        f"ROUGE-L F1: {avg['rougeL']:.4f}"
    )
    if args.output:
        print(f"Wrote summary to {args.output}")
    else:
        print(json.dumps(summary["average"], indent=2))


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if os.path.isdir(args.input):
        _run_directory_mode(args)
    else:
        _run_file_mode(args)


if __name__ == "__main__":
    main()
