from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import csv
from dotenv import load_dotenv

from .entity_extract import extract_entities_for_validation
from .judge import judge_against_ground_truth
from .metrics import entity_recall, structural_alignment
from .schemas import GTValidatorReport
from .section_parser import parse_soap_sections


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
    """
    Best-effort check: whether an omitted item is actually present in the transcript.

    Returns:
    - True: transcript appears to contain the omitted content
    - False: transcript likely does not contain it (i.e., omission is GT-only / fabricated)
    - None: cannot determine (missing transcript or too little signal)
    """
    t = (transcript or "").strip().lower()
    if not t:
        return None
    o = (omission or "").strip().lower()
    if not o:
        return None

    # If omission contains a quoted phrase, prefer matching that directly.
    m = re.search(r"'([^']{4,})'", omission)
    if not m:
        m = re.search(r"\"([^\"]{4,})\"", omission)
    if m:
        phrase = m.group(1).strip().lower()
        if phrase and phrase in t:
            return True
        # Phrase not found: still might be paraphrased; fall through.

    # Token match heuristic: require >=2 informative tokens to be present.
    toks = re.findall(r"[a-z0-9][a-z0-9\\-]{2,}", o)
    toks = [x for x in toks if x not in _OMISSION_STOP and len(x) >= 3]
    toks = list(dict.fromkeys(toks))  # preserve order, unique
    if len(toks) < 2:
        return None
    hit = sum(1 for x in toks if x in t)
    if hit >= 2:
        return True
    if hit == 0:
        return False
    return None


def load_env() -> None:
    """Load `.env` from repo root if present."""
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")

    # Normalize to GOOGLE_API_KEY if user stored one of the common aliases.
    key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("gemini_api_key")
    )
    if key:
        os.environ.setdefault("GOOGLE_API_KEY", key)


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    analysis_dir: Path
    limit: int | None = None
    skip_llm: bool = False
    judge_model: str | None = None
    evidence_model: str | None = None
    allow_evidence_gemini_fallback: bool = True


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _iter_item_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output dir not found: {output_dir}")
    return sorted([p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("item_")])


def run_one_item(
    item_dir: Path,
    *,
    skip_llm: bool,
    judge_model: str | None,
    evidence_model: str | None = None,
) -> GTValidatorReport:
    source_path = item_dir / "source.json"
    a3_path = item_dir / "agent3_result.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing {source_path}")
    if not a3_path.is_file():
        raise FileNotFoundError(f"Missing {a3_path}")

    source = _read_json(source_path)
    a3 = _read_json(a3_path)

    gt = str(source.get("ground_truth") or "")
    gen = str(a3.get("generated") or "")
    transcript = str(a3.get("transcript") or "")
    agent3_claims = ((a3.get("claim_verification") or {}) or {}).get("claims") or []
    if not gt.strip():
        raise ValueError(f"Empty ground_truth in {source_path}")
    if not gen.strip():
        raise ValueError(f"Empty generated SOAP in {a3_path} (expected key: generated)")
    if not transcript.strip():
        # Not fatal for GT-vs-generated metrics, but required for transcript evidence checks.
        transcript = ""

    ent_gt = extract_entities_for_validation(gt)
    ent_gen = extract_entities_for_validation(gen)

    recall_all = entity_recall(ent_gt.all_entities, ent_gen.all_entities)
    recall_sym = entity_recall(ent_gt.symptoms, ent_gen.symptoms)
    recall_med = entity_recall(ent_gt.medications, ent_gen.medications)
    recall_lab = entity_recall(ent_gt.labs, ent_gen.labs)

    gt_sections = parse_soap_sections(gt)
    gen_sections = parse_soap_sections(gen)
    align = structural_alignment(gt_sections, gen_sections)

    judge = None
    judge_err = None
    addition_evidence = []
    if not skip_llm:
        try:
            judge, addition_evidence = judge_against_ground_truth(
                ground_truth_soap=gt,
                generated_soap=gen,
                transcript=transcript,
                model=judge_model,
                evidence_model=evidence_model,
                agent3_claims=agent3_claims if isinstance(agent3_claims, list) else [],
                allow_evidence_gemini_fallback=True,
            )
        except Exception as e:
            judge = None
            judge_err = str(e)

    return GTValidatorReport(
        item=item_dir.name,
        output_dir=str(item_dir),
        ground_truth_soap=gt,
        generated_soap=gen,
        entities_gt=ent_gt,
        entities_generated=ent_gen,
        clinical_recall_all=recall_all,
        clinical_recall_symptoms=recall_sym,
        clinical_recall_medications=recall_med,
        clinical_recall_labs=recall_lab,
        structural_alignment=align,
        expert_judge=judge,
        expert_judge_error=judge_err,
        addition_evidence=addition_evidence,
    )


def run_batch(cfg: RunConfig) -> int:
    load_env()

    item_dirs = _iter_item_dirs(cfg.output_dir)
    if cfg.limit is not None:
        item_dirs = item_dirs[: cfg.limit]
    if not item_dirs:
        raise RuntimeError(f"No item_* folders found under {cfg.output_dir}")

    rows: list[dict] = []
    for item_dir in item_dirs:
        print(f"Running {item_dir}")
        # Inline run_one_item for access to cfg knobs without broad signature churn.
        report = run_one_item(
            item_dir,
            skip_llm=cfg.skip_llm,
            judge_model=cfg.judge_model,
            evidence_model=cfg.evidence_model,
        )
        out_path = item_dir / "gt_validator.json"
        _write_json(out_path, report.model_dump())

        n_sup = n_unsup = n_unk = None
        if report.addition_evidence:
            n_sup = sum(1 for a in report.addition_evidence if a.supported_by_transcript == "supported")
            n_unsup = sum(1 for a in report.addition_evidence if a.supported_by_transcript == "unsupported")
            n_unk = sum(1 for a in report.addition_evidence if a.supported_by_transcript == "unknown")

        n_additions_total = None
        n_omissions_total = None
        n_omissions_unsupported = None
        if report.expert_judge:
            n_additions_total = len(report.expert_judge.additions or [])
            n_omissions_total = len(report.expert_judge.omissions or [])

        # If we have the transcript (via judge inputs) we can approximate which omissions
        # are GT-only (unsupported by transcript).
        if report.expert_judge and report.generated_soap:
            # transcript lives in agent3_result.json; reuse by re-reading to avoid schema changes
            # (keeps report stable). If missing, this yields None.
            a3_path = (item_dir / "agent3_result.json")
            try:
                a3 = _read_json(a3_path)
                transcript = str(a3.get("transcript") or "")
            except Exception:
                transcript = ""
            if transcript and report.expert_judge.omissions:
                unsupported = 0
                for om in report.expert_judge.omissions:
                    ok = _omission_supported_by_transcript(str(om or ""), transcript)
                    if ok is False:
                        unsupported += 1
                n_omissions_unsupported = unsupported

        row = {
            "item": report.item,
            "clinical_recall_all": report.clinical_recall_all.recall,
            "clinical_recall_symptoms": report.clinical_recall_symptoms.recall,
            "clinical_recall_medications": report.clinical_recall_medications.recall,
            "clinical_recall_labs": report.clinical_recall_labs.recall,
            "overall_token_f1": report.structural_alignment.overall_token_f1,
            "overall_rouge_l_f1": report.structural_alignment.overall_rouge_l_f1,
            "assessment_alignment_score": report.structural_alignment.assessment_alignment_score,
            "plan_alignment_score": report.structural_alignment.plan_alignment_score,
            "judge_overall_grade": (report.expert_judge.overall_grade if report.expert_judge else None),
            "judge_error": report.expert_judge_error,
            "n_omissions": n_omissions_total,
            "n_additions": n_additions_total,
            "n_additions_supported_by_transcript": n_sup,
            "n_additions_unsupported_by_transcript": n_unsup,
            "n_additions_unknown_by_transcript": n_unk,
            "n_omissions_unsupported_by_transcript": n_omissions_unsupported,
        }
        rows.append(row)

    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.analysis_dir / "gt_validator_summary.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return len(rows)

