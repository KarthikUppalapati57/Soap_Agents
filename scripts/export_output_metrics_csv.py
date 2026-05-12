"""
Flatten Soap_Agents/Output/item_* into one CSV (one row per item).

Includes GT validator metrics (omissions, unsupported additions, clinical recall,
structural alignment) and Agent 3 benchmark / claim-verification aggregates.

Run from Soap_Agents:
  uv run python scripts/export_output_metrics_csv.py

Default output: Output/analysis/output_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# Mirrors Agent_GT_Validator.runner._omission_supported_by_transcript (avoid importing runner → google.genai).
_OMISSION_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "with",
    "on",
    "as",
    "is",
    "are",
    "was",
    "were",
    "patient",
    "patients",
    "report",
    "reports",
    "denies",
    "history",
    "assessment",
    "objective",
    "subjective",
    "plan",
    "icd",
    "icd-10",
    "code",
}


def _omission_supported_by_transcript(omission: str, transcript: str) -> bool | None:
    t = (transcript or "").strip().lower()
    if not t:
        return None
    o = (omission or "").strip().lower()
    if not o:
        return None
    m = re.search(r"'([^']{4,})'", omission)
    if not m:
        m = re.search(r'"([^"]{4,})"', omission)
    if m:
        phrase = m.group(1).strip().lower()
        if phrase and phrase in t:
            return True
    toks = re.findall(r"[a-z0-9][a-z0-9\-]{2,}", o)
    toks = [x for x in toks if x not in _OMISSION_STOP and len(x) >= 3]
    toks = list(dict.fromkeys(toks))
    if len(toks) < 2:
        return None
    hit = sum(1 for x in toks if x in t)
    if hit >= 2:
        return True
    if hit == 0:
        return False
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _iter_item_dirs(output_dir: Path) -> list[Path]:
    return sorted(p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("item_"))


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _agent3_row(folder: Path) -> dict[str, object]:
    path = folder / "agent3_result.json"
    d = _read_json(path)
    if not d:
        return {"has_agent3_result": False}
    row: dict[str, object] = {"has_agent3_result": True}
    skip = {"transcript", "generated", "claim_verification"}
    for k, v in d.items():
        if k in skip:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            row[f"agent3_{k}"] = v
    cv = d.get("claim_verification") or {}
    claims = cv.get("claims") or []
    n_total = len(claims) if isinstance(claims, list) else 0
    n_unsup = sum(
        1 for c in claims if isinstance(c, dict) and c.get("support_status") == "unsupported"
    )
    row["agent3_n_claims"] = n_total
    row["agent3_n_unsupported_claims"] = n_unsup
    row["agent3_unsupported_claim_rate"] = (n_unsup / n_total) if n_total else float("nan")
    return row


def _meta_row(folder: Path) -> dict[str, object]:
    d = _read_json(folder / "meta.json")
    if not d:
        return {}
    return {f"meta_{k}": v for k, v in d.items() if isinstance(v, (str, int, float, bool)) or v is None}


def _flatten_clinical_recall(prefix: str, obj: object) -> dict[str, object]:
    if not isinstance(obj, dict):
        return {}
    out: dict[str, object] = {}
    for k, v in obj.items():
        if k in ("gt_only", "gen_only", "overlap") and isinstance(v, list):
            out[f"{prefix}_n_{k}"] = len(v)
        elif isinstance(v, (int, float, bool)) or v is None:
            out[f"{prefix}_{k}"] = v
    return out


def _flatten_structural(obj: object) -> dict[str, object]:
    if not isinstance(obj, dict):
        return {}
    out: dict[str, object] = {}
    for k in ("overall_token_f1", "overall_rouge_l_f1", "assessment_alignment_score", "plan_alignment_score"):
        if k in obj:
            out[f"structural_{k}"] = obj[k]
    sections = obj.get("by_section")
    if isinstance(sections, list):
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            name = str(sec.get("section") or "unknown").replace(" ", "_")
            for fld in ("gt_present", "gen_present", "token_f1", "rouge_l_f1"):
                if fld in sec:
                    out[f"structural_section_{name}_{fld}"] = sec[fld]
    return out


def _flatten_entities(label: str, obj: object) -> dict[str, object]:
    if not isinstance(obj, dict):
        return {}
    out: dict[str, object] = {}
    for k, v in obj.items():
        if isinstance(v, list):
            out[f"entities_{label}_n_{k}"] = len(v)
    return out


def _gt_validator_row(folder: Path, transcript: str) -> dict[str, object]:
    d = _read_json(folder / "gt_validator.json")
    if not d:
        return {"has_gt_validator": False}
    row: dict[str, object] = {"has_gt_validator": True}
    row["gt_item"] = d.get("item")
    row["gt_validator_error"] = d.get("expert_judge_error")

    ej = d.get("expert_judge")
    if isinstance(ej, dict):
        row["judge_overall_grade"] = ej.get("overall_grade")
        row["judge_overall_summary"] = ej.get("overall_summary")
        omissions = ej.get("omissions") or []
        additions = ej.get("additions") or []
        hall = ej.get("hallucinations_or_unjustified_inferences") or []
        row["n_omissions"] = len(omissions) if isinstance(omissions, list) else None
        row["n_additions"] = len(additions) if isinstance(additions, list) else None
        row["n_hallucinations_or_unjustified_inferences"] = len(hall) if isinstance(hall, list) else None
        n_om_unsup: int | None = None
        if isinstance(omissions, list):
            if not omissions:
                n_om_unsup = 0
            elif transcript:
                n_om_unsup = 0
                for om in omissions:
                    ok = _omission_supported_by_transcript(str(om or ""), transcript)
                    if ok is False:
                        n_om_unsup += 1
        row["n_omissions_unsupported_by_transcript"] = n_om_unsup

        sec_grades = ej.get("section_grades")
        if isinstance(sec_grades, list):
            n_sec_om = n_sec_add = 0
            for sg in sec_grades:
                if not isinstance(sg, dict):
                    continue
                o = sg.get("omissions") or []
                a = sg.get("additions") or []
                if isinstance(o, list):
                    n_sec_om += len(o)
                if isinstance(a, list):
                    n_sec_add += len(a)
            row["n_section_grades_omissions"] = n_sec_om
            row["n_section_grades_additions"] = n_sec_add

    ev = d.get("addition_evidence")
    n_sup = n_unsup = n_unk = None
    n_pk_sup = n_pk_unsup = n_pk_unk = None
    n_tr_or_pk_sup = n_tr_unsup_pk_sup = n_both_unsup = None
    if isinstance(ev, list) and ev:
        n_sup = sum(1 for a in ev if isinstance(a, dict) and a.get("supported_by_transcript") == "supported")
        n_unsup = sum(1 for a in ev if isinstance(a, dict) and a.get("supported_by_transcript") == "unsupported")
        n_unk = sum(1 for a in ev if isinstance(a, dict) and a.get("supported_by_transcript") == "unknown")
        n_pk_sup = sum(1 for a in ev if isinstance(a, dict) and a.get("supported_by_primekg") == "supported")
        n_pk_unsup = sum(1 for a in ev if isinstance(a, dict) and a.get("supported_by_primekg") == "unsupported")
        n_pk_unk = sum(
            1
            for a in ev
            if isinstance(a, dict)
            and a.get("supported_by_primekg") in ("unknown", None)
        )
        n_tr_or_pk_sup = sum(
            1
            for a in ev
            if isinstance(a, dict)
            and (
                a.get("supported_by_transcript") == "supported"
                or a.get("supported_by_primekg") == "supported"
            )
        )
        n_tr_unsup_pk_sup = sum(
            1
            for a in ev
            if isinstance(a, dict)
            and a.get("supported_by_transcript") == "unsupported"
            and a.get("supported_by_primekg") == "supported"
        )
        n_both_unsup = sum(
            1
            for a in ev
            if isinstance(a, dict)
            and a.get("supported_by_transcript") == "unsupported"
            and a.get("supported_by_primekg") == "unsupported"
        )
    row["n_addition_evidence_entries"] = len(ev) if isinstance(ev, list) else None
    row["n_additions_supported_by_transcript"] = n_sup
    row["n_additions_unsupported_by_transcript"] = n_unsup
    row["n_additions_unknown_by_transcript"] = n_unk
    row["n_additions_supported_by_primekg"] = n_pk_sup
    row["n_additions_unsupported_by_primekg"] = n_pk_unsup
    row["n_additions_unknown_by_primekg"] = n_pk_unk
    row["n_additions_supported_by_transcript_or_primekg"] = n_tr_or_pk_sup
    row["n_additions_transcript_unsupported_primekg_supported"] = n_tr_unsup_pk_sup
    row["n_additions_unsupported_by_transcript_and_primekg"] = n_both_unsup

    for key in list(d.keys()):
        if key.startswith("clinical_recall_") and isinstance(d[key], dict):
            prefix = key
            row.update(_flatten_clinical_recall(prefix, d[key]))

    row.update(_flatten_structural(d.get("structural_alignment")))
    row.update(_flatten_entities("gt", d.get("entities_gt")))
    row.update(_flatten_entities("generated", d.get("entities_generated")))

    return row


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Export per-item metrics from Output/ to CSV.")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Output",
        help="Path to Output directory (default: Soap_Agents/Output).",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default="output_metrics.csv",
        help="Output CSV path.",
    )
    args = ap.parse_args()
    output_dir: Path = args.output_dir
    out_csv: Path = args.csv
    out_csv = output_dir / "analysis" / out_csv

    rows: list[dict[str, object]] = []
    for folder in _iter_item_dirs(output_dir):
        a3_path = folder / "agent3_result.json"
        a3_raw = _read_json(a3_path)
        transcript = str((a3_raw or {}).get("transcript") or "")
        row: dict[str, object] = {"item": folder.name}
        row.update(_meta_row(folder))
        row.update(_agent3_row(folder))
        row.update(_gt_validator_row(folder, transcript))
        rows.append(row)

    if not rows:
        raise SystemExit(f"No item_* folders under {output_dir}")

    df = pd.DataFrame(rows)
    # Stable column order: item + meta + key metrics + rest sorted
    preferred = [
        "item",
        "meta_index",
        "meta_iterations",
        "has_agent3_result",
        "has_gt_validator",
        "n_omissions",
        "n_additions",
        "n_additions_unsupported_by_transcript",
        "n_additions_supported_by_transcript",
        "n_additions_unknown_by_transcript",
        "n_additions_supported_by_primekg",
        "n_additions_unsupported_by_primekg",
        "n_additions_unknown_by_primekg",
        "n_additions_supported_by_transcript_or_primekg",
        "n_additions_transcript_unsupported_primekg_supported",
        "n_additions_unsupported_by_transcript_and_primekg",
        "n_omissions_unsupported_by_transcript",
        "n_hallucinations_or_unjustified_inferences",
        "agent3_n_claims",
        "agent3_n_unsupported_claims",
        "agent3_unsupported_claim_rate",
        "judge_overall_grade",
    ]
    cols = [c for c in preferred if c in df.columns]
    rest = sorted(c for c in df.columns if c not in cols)
    df = df[cols + rest]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
