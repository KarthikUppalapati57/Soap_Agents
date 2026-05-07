from __future__ import annotations

import json
import os
from typing import Any

from google.genai import Client, types
from pydantic import ValidationError

from .addition_evidence import build_addition_evidence
from .schemas import (
    AdditionEvidence,
    ExpertJudgeDiscrepancies,
    ExpertJudgeGradingResult,
    ExpertJudgeReport,
    JudgeSectionGrade,
    JudgeSectionGradeOutcome,
)


def _default_model() -> str:
    return os.getenv("GTJUDGE_GEMINI_MODEL", "gemini-2.5-flash")


def _max_tokens() -> int | None:
    v = (os.getenv("MAX_TOKENS") or "").strip()
    if not v:
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    return n if n > 0 else None


_RUBRIC_COMPARE = """Rubric (list omissions/additions only — do NOT assign letter grades):

Primary objective:
- Compare Generated SOAP vs GroundTruth SOAP and identify differences:
  - omissions: items present in GroundTruthSOAP that are entirely missing from GeneratedSOAP.
  - additions: items present in GeneratedSOAP that are not present in GroundTruthSOAP.

Hard rules:
- Omission means ENTIRELY MISSING. If a concept is present in GeneratedSOAP but worded differently, DO NOT call it an omission.
- If GeneratedSOAP adds qualifiers/details not present in GroundTruthSOAP, those are additions (possibly over-specific), not omissions.
- Only compare GroundTruthSOAP vs GeneratedSOAP in this step. Ignore Transcript for deciding omissions/additions.
- Do NOT list additions that are purely formatting/notation changes, including common abbreviations or acronyms
  that preserve the same meaning (e.g. "urinary tract infection" vs "(UTI)", "blood in urine" vs "hematuria",
  or frequency abbreviations like BID/PRN when the underlying instruction is the same).

Be concrete: each omission/addition should be a short bullet-like string that can be checked against the texts.
Avoid “missing exact phrasing” type omissions; focus on missing content.

Include one section_discrepancies entry for each of: Subjective, Objective, Assessment, Plan (use empty lists when nothing to report).
"""


_RUBRIC_GRADE = """Rubric (grading ONLY from omissions and unsupported/unknown additions):

You receive:
1) The same GroundTruthSOAP, GeneratedSOAP, and Transcript as before (for context when writing summaries).
2) discrepancy_lists: omissions, additions, and per-section omissions/additions from a prior review step.
3) addition_evidence: for many additions, transcript support status from an automated check (supported | unsupported | unknown).

Grading inputs (ONLY these may lower a grade):
- All omissions in discrepancy_lists (GT-fabricated items were already excluded in the prior step).
- Additions whose addition_evidence.supported_by_transcript is **unsupported** or **unknown**.

Severity note:
- Treat unsupported additions as high-severity alignment issues by default, especially in Assessment/Plan.
- Even a small number of unsupported additions can justify dropping below B if they introduce misleading clinical content.

**Supported additions:** additions with supported_by_transcript == **supported** must NOT reduce any section or overall grade. They may appear in the text for transparency but are clinically grounded in the Transcript.

If an addition has no entry in addition_evidence, treat it as **unknown** for grading (conservative).

How to weigh issues:
- Focus on clinical materiality and usefulness.
- A small number of minor omissions or minor unsupported/unknown additions should not automatically drop the grade.
- Prefer “benefit of the doubt” when deciding severity unless the issue clearly changes meaning, safety, or clinical usefulness.
- If the note is broadly clinically usable and the issues are limited in scope, avoid over-penalizing.

Letter scale (based ONLY on the grading inputs above):
- Grade A: captures all major clinically important GT content; any omissions are minor; unsupported/unknown additions are absent or minor.
- Grade B: mostly clinically aligned and clearly usable; may have several minor issues and/or up to 1–2 moderate issues.
- Grade C: noticeable gaps or multiple moderate issues that make the note meaningfully incomplete; still usable with caution and review.
- Grade D: major issues that materially reduce clinical usefulness or introduce misleading content.
- Grade F: unsafe/unreliable; severe unjustified content or missing critical GT content.

Be concrete in summaries: reference omissions and unsupported/unknown additions, not supported extras.
"""


def _prompt_discrepancies(ground_truth: str, generated: str, transcript: str) -> str:
    return f"""You are an expert clinical documentation reviewer.

You will be given:
1) GroundTruthSOAP: the reference note
2) GeneratedSOAP: the model-generated note to evaluate against the reference
3) Transcript: the source conversation (provided for later steps; ignore it for omissions/additions here)

Task:
- Compare GeneratedSOAP to GroundTruthSOAP.
- Identify omissions and additions (global and per SOAP section).
- Do NOT assign letter grades in this step.

Important rule about GT-fabricated fields:
- You will also receive the Transcript (source conversation).
- Some GroundTruthSOAP fields may be fabricated/templated and not present in the Transcript.
- Do NOT include these as omissions:
  - patient name
  - patient age or DOB
  - sex/gender
  - race/ethnicity
  - ICD-10 codes (or other billing/coding identifiers)

Important constraints:
- Use only the two SOAP texts for omissions/additions. (Transcript is irrelevant in this step.)
- Return ONLY valid JSON matching the required schema.
- Do not create “omissions” based on phrasing differences—only entirely missing items.

Required JSON schema:
{{
  "model": "<string model name>",
  "omissions": ["<string>", ...],
  "additions": ["<string>", ...],
  "section_discrepancies": [
    {{
      "section": "Subjective|Objective|Assessment|Plan",
      "omissions": ["<string>", ...],
      "additions": ["<string>", ...]
    }}
  ],
  "rubric_notes": "<optional string or null>"
}}

{_RUBRIC_COMPARE}

GroundTruthSOAP:
\"\"\"{ground_truth}\"\"\"

GeneratedSOAP:
\"\"\"{generated}\"\"\"

Transcript:
\"\"\"{transcript}\"\"\"
"""


def _prompt_grading(
    ground_truth: str,
    generated: str,
    transcript: str,
    discrepancy_json: str,
    addition_evidence_json: str,
) -> str:
    return f"""You are an expert clinical documentation reviewer assigning letter grades.

{_RUBRIC_GRADE}

GroundTruthSOAP:
\"\"\"{ground_truth}\"\"\"

GeneratedSOAP:
\"\"\"{generated}\"\"\"

Transcript:
\"\"\"{transcript}\"\"\"

discrepancy_lists (JSON):
{discrepancy_json}

addition_evidence (JSON):
{addition_evidence_json}

Task:
- Assign overall_grade and overall_summary.
- Assign per-section grade and summary for each section in section_discrepancies (same section names).
- Apply the rubric using ONLY omissions and unsupported/unknown additions as defined above.

Important constraints:
- Do NOT use outside medical knowledge.
- Return ONLY valid JSON matching the required schema.

Required JSON schema:
{{
  "model": "<string model name>",
  "overall_grade": "A|B|C|D|F",
  "overall_summary": "<short paragraph>",
  "section_grades": [
    {{
      "section": "Subjective|Objective|Assessment|Plan",
      "grade": "A|B|C|D|F",
      "summary": "<short paragraph>"
    }}
  ],
  "rubric_notes": "<optional string or null>"
}}
"""


def _unique_additions_in_order(disc: ExpertJudgeDiscrepancies) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for a in disc.additions:
        t = str(a or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    for sec in disc.section_discrepancies:
        for a in sec.additions:
            t = str(a or "").strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _merge_expert_report(
    disc: ExpertJudgeDiscrepancies,
    grading: ExpertJudgeGradingResult,
    *,
    model_fallback: str,
) -> ExpertJudgeReport:
    by_section: dict[str, JudgeSectionGradeOutcome] = {
        str(g.section): g for g in grading.section_grades
    }
    section_grades: list[JudgeSectionGrade] = []
    for s in disc.section_discrepancies:
        g = by_section.get(str(s.section))
        if g is None:
            raise RuntimeError(
                f"Grading step missing section {s.section!r}; got {list(by_section.keys())}"
            )
        section_grades.append(
            JudgeSectionGrade(
                section=s.section,
                grade=g.grade,
                summary=g.summary,
                omissions=s.omissions,
                additions=s.additions,
            )
        )

    notes_parts = [disc.rubric_notes, grading.rubric_notes]
    merged_notes = " | ".join(str(n).strip() for n in notes_parts if n and str(n).strip()) or None

    return ExpertJudgeReport(
        model=(grading.model or disc.model or model_fallback).strip() or model_fallback,
        overall_grade=grading.overall_grade,
        overall_summary=grading.overall_summary,
        omissions=disc.omissions,
        additions=disc.additions,
        hallucinations_or_unjustified_inferences=[],
        section_grades=section_grades,
        rubric_notes=merged_notes,
    )


def _generate_json(model: str, prompt: str, *, temperature: float) -> dict[str, Any]:
    client = Client()
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=temperature,
            max_output_tokens=_max_tokens(),
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Judge returned empty response text.")
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Judge returned non-JSON output: {e}") from e
    if not isinstance(obj, dict):
        raise RuntimeError("Judge returned JSON that is not an object.")
    return obj


def judge_list_discrepancies(
    *,
    ground_truth_soap: str,
    generated_soap: str,
    transcript: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> ExpertJudgeDiscrepancies:
    m = model or _default_model()
    obj = _generate_json(m, _prompt_discrepancies(ground_truth_soap, generated_soap, transcript), temperature=temperature)
    obj.setdefault("model", m)
    try:
        return ExpertJudgeDiscrepancies.model_validate(obj)
    except ValidationError as e:
        raise RuntimeError(f"Judge JSON did not match discrepancies schema: {e}") from e


def judge_assign_grades(
    *,
    discrepancies: ExpertJudgeDiscrepancies,
    addition_evidence: list[AdditionEvidence],
    ground_truth_soap: str,
    generated_soap: str,
    transcript: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> ExpertJudgeGradingResult:
    m = model or _default_model()
    disc_json = json.dumps(discrepancies.model_dump(exclude={"model"}), ensure_ascii=False, indent=2)
    ev_json = json.dumps([e.model_dump() for e in addition_evidence], ensure_ascii=False, indent=2)
    obj = _generate_json(
        m,
        _prompt_grading(ground_truth_soap, generated_soap, transcript, disc_json, ev_json),
        temperature=temperature,
    )
    obj.setdefault("model", m)
    try:
        return ExpertJudgeGradingResult.model_validate(obj)
    except ValidationError as e:
        raise RuntimeError(f"Judge JSON did not match grading schema: {e}") from e


def judge_against_ground_truth(
    *,
    ground_truth_soap: str,
    generated_soap: str,
    transcript: str,
    model: str | None = None,
    temperature: float = 0.0,
    evidence_model: str | None = None,
    agent3_claims: list[dict[str, Any]] | None = None,
    allow_evidence_gemini_fallback: bool = True,
) -> tuple[ExpertJudgeReport, list[AdditionEvidence]]:
    """
    Two-phase expert judge:
    1) List omissions/additions vs GT.
    2) Classify transcript support for deduped additions, then assign grades using only
       omissions + unsupported/unknown additions.
    """
    m = model or _default_model()
    disc = judge_list_discrepancies(
        ground_truth_soap=ground_truth_soap,
        generated_soap=generated_soap,
        transcript=transcript,
        model=m,
        temperature=temperature,
    )
    adds = _unique_additions_in_order(disc)
    t = (transcript or "").strip()
    if adds and t:
        addition_evidence = build_addition_evidence(
            additions=adds,
            transcript=transcript,
            agent3_claims=agent3_claims if isinstance(agent3_claims, list) else [],
            allow_gemini_fallback=allow_evidence_gemini_fallback,
            gemini_model=evidence_model,
        )
    elif adds:
        addition_evidence = [
            AdditionEvidence(
                addition_text=a,
                supported_by_transcript="unknown",
                transcript_evidence=None,
                evidence_source="none",
                matched_claim_text=None,
            )
            for a in adds
        ]
    else:
        addition_evidence = []

    grading = judge_assign_grades(
        discrepancies=disc,
        addition_evidence=addition_evidence,
        ground_truth_soap=ground_truth_soap,
        generated_soap=generated_soap,
        transcript=transcript,
        model=m,
        temperature=temperature,
    )
    report = _merge_expert_report(disc, grading, model_fallback=m)
    # Hallucinations are defined purely as transcript-unsupported additions.
    report.hallucinations_or_unjustified_inferences = [
        a.addition_text for a in addition_evidence if a.supported_by_transcript == "unsupported"
    ]
    return report, addition_evidence


# Backwards compatibility for tests: single-discrepancies prompt shape.
def _prompt(ground_truth: str, generated: str, transcript: str) -> str:
    return _prompt_discrepancies(ground_truth, generated, transcript)
