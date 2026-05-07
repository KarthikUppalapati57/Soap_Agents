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
- Compare Generated SOAP vs GroundTruth SOAP and identify:
  - omissions: clinically important items present in GroundTruth but missing/less specific in Generated
  - additions: items in Generated not present in GroundTruth (including over-specific claims)

Important rule about GT-fabricated fields:
- You will also receive the Transcript (source conversation).
- Some GroundTruthSOAP fields may be fabricated/templated and not present in the Transcript.
- Do NOT include these as omissions:
  - patient name
  - patient age or DOB
  - sex/gender
  - race/ethnicity
  - ICD-10 codes (or other billing/coding identifiers)

Be concrete: each omission/addition should be a short bullet-like string that can be checked against the texts.
If something is partially present, prefer listing it as an omission with what detail is missing.

Include one section_discrepancies entry for each of: Subjective, Objective, Assessment, Plan, Other (use empty lists when nothing to report).
"""


_RUBRIC_GRADE = """Rubric (grading ONLY from omissions, unsupported/unknown additions, and hallucinations):

You receive:
1) The same GroundTruthSOAP, GeneratedSOAP, and Transcript as before (for context when writing summaries).
2) discrepancy_lists: omissions, additions, per-section omissions/additions, and hallucinations_or_unjustified_inferences from a prior review step.
3) addition_evidence: for many additions, transcript support status from an automated check (supported | unsupported | unknown).

Grading inputs (ONLY these may lower a grade):
- All omissions in discrepancy_lists (GT-fabricated items were already excluded in the prior step).
- Additions whose addition_evidence.supported_by_transcript is **unsupported** or **unknown**.
- Entries in hallucinations_or_unjustified_inferences (treat as severe alignment/safety issues).

**Supported additions:** additions with supported_by_transcript == **supported** must NOT reduce any section or overall grade. They may appear in the text for transparency but are clinically grounded in the Transcript.

If an addition has no entry in addition_evidence, treat it as **unknown** for grading (conservative).

Letter scale (based ONLY on the grading inputs above):
- Grade A: essentially all clinically important GT content captured; negligible unsupported/unknown additions; hallucination list empty or trivial.
- Grade B: minor issues among the grading inputs, but overall clinically aligned.
- Grade C: multiple grading-input issues; some sections weaker but still usable.
- Grade D: major grading-input issues that reduce clinical usefulness.
- Grade F: unsafe/unreliable; pervasive unjustified content or missing critical GT content.

Be concrete in summaries: reference omissions and unsupported/unknown additions, not supported extras.
"""


def _prompt_discrepancies(ground_truth: str, generated: str, transcript: str) -> str:
    return f"""You are an expert clinical documentation reviewer.

You will be given:
1) GroundTruthSOAP: the reference note
2) GeneratedSOAP: the model-generated note to evaluate against the reference
3) Transcript: the source conversation (used ONLY to decide whether certain GT fields are fabricated)

Task:
- Compare GeneratedSOAP to GroundTruthSOAP.
- Identify omissions and additions (global and per SOAP section).
- Identify hallucinations_or_unjustified_inferences (Generated claims not justified by GroundTruthSOAP or Transcript).
- Do NOT assign letter grades in this step.

Important constraints:
- Do NOT use outside medical knowledge. Only compare the two texts (and Transcript only for the GT-fabricated omission rule).
- Return ONLY valid JSON matching the required schema.
- For omissions: apply the GT-fabricated-fields rule from the rubric using the Transcript.

Required JSON schema:
{{
  "model": "<string model name>",
  "omissions": ["<string>", ...],
  "additions": ["<string>", ...],
  "hallucinations_or_unjustified_inferences": ["<string>", ...],
  "section_discrepancies": [
    {{
      "section": "Subjective|Objective|Assessment|Plan|Other",
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
- Apply the rubric using ONLY omissions, unsupported/unknown additions, and hallucinations as defined above.

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
      "section": "Subjective|Objective|Assessment|Plan|Other",
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
    addition_evidence: list[AdditionEvidence],
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
        hallucinations_or_unjustified_inferences=disc.hallucinations_or_unjustified_inferences,
        section_grades=section_grades,
        rubric_notes=merged_notes,
        addition_evidence=addition_evidence,
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
) -> ExpertJudgeReport:
    """
    Two-phase expert judge:
    1) List omissions/additions (and hallucinations) vs GT.
    2) Classify transcript support for deduped additions, then assign grades using only
       omissions + unsupported/unknown additions + hallucinations.
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
    return _merge_expert_report(disc, grading, addition_evidence, model_fallback=m)


# Backwards compatibility for tests: single-discrepancies prompt shape.
def _prompt(ground_truth: str, generated: str, transcript: str) -> str:
    return _prompt_discrepancies(ground_truth, generated, transcript)
