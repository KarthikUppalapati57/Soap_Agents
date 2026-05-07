from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

from google.genai import Client, types

from .schemas import AdditionEvidence, TranscriptSupport


_TOK_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

_ACRONYM_EXPANSIONS: dict[str, list[str]] = {
    # Common clinical acronyms we see in SOAP notes.
    "uti": ["urinary tract infection"],
}

_TERM_SYNONYMS: dict[str, list[str]] = {
    # Prefer simple, unambiguous paraphrases.
    "hematuria": ["blood in urine", "blood in the urine"],
}


def _tokens(s: str) -> set[str]:
    return set(_TOK_RE.findall((s or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


def _best_claim_match(addition: str, claims: list[dict[str, Any]]) -> dict[str, Any] | None:
    add = (addition or "").strip()
    if not add:
        return None
    add_toks = _tokens(add)

    best = None
    best_score = 0.0
    for cl in claims:
        ct = str(cl.get("claim_text") or "").strip()
        if not ct:
            continue
        ct_l = ct.lower()
        add_l = add.lower()
        # Fast path: substring either way.
        if add_l in ct_l or ct_l in add_l:
            return cl
        score = _jaccard(add_toks, _tokens(ct))
        if score > best_score:
            best_score = score
            best = cl

    # Threshold tuned to avoid spurious matches on short phrases.
    return best if best is not None and best_score >= 0.22 else None


def _evidence_from_claim(addition: str, cl: dict[str, Any]) -> AdditionEvidence:
    st = str(cl.get("support_status") or "").strip().lower()
    if st == "supported":
        supported: TranscriptSupport = "supported"
    elif st == "unsupported":
        supported = "unsupported"
    else:
        supported = "unknown"
    ev = cl.get("transcript_evidence")
    ev_str = str(ev).strip() if isinstance(ev, str) and ev.strip() else None
    return AdditionEvidence(
        addition_text=addition,
        supported_by_transcript=supported,
        transcript_evidence=ev_str,
        evidence_source="agent3_claims",
        matched_claim_text=str(cl.get("claim_text") or "").strip() or None,
    )


def _default_model() -> str:
    # Default to the same model family as the judge for consistency.
    return os.getenv("GTV_EVIDENCE_GEMINI_MODEL", os.getenv("GTJUDGE_GEMINI_MODEL", "gemini-2.5-flash"))


def _max_tokens() -> int | None:
    v = (os.getenv("MAX_TOKENS") or "").strip()
    if not v:
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    return n if n > 0 else None


def _gemini_prompt(transcript: str, addition: str) -> str:
    return f"""You are checking whether a single statement is supported by a transcript.

Inputs:
- Transcript: source conversation.
- Addition: a statement extracted from a generated SOAP note that was not in the ground-truth SOAP.

Task:
- Determine if the Addition is clearly supported by the Transcript.
- If supported, return a short verbatim quote (or tight paraphrase) from the Transcript as evidence.

Hard rules:
- Use ONLY the Transcript. No outside knowledge.
- If the Transcript does not clearly support the Addition, mark unsupported.
- Return ONLY JSON.

Required JSON schema:
{{
  "supported_by_transcript": "supported|unsupported",
  "transcript_evidence": "<string or null>",
  "rationale": "<short string>"
}}

Transcript:
\"\"\"{transcript}\"\"\"

Addition:
\"\"\"{addition}\"\"\"
"""


def _gemini_check(transcript: str, addition: str, *, model: str | None = None) -> AdditionEvidence:
    m = model or _default_model()
    client = Client()
    resp = client.models.generate_content(
        model=m,
        contents=_gemini_prompt(transcript, addition),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=_max_tokens(),
        ),
    )
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    obj = json.loads(text)
    sup = str(obj.get("supported_by_transcript") or "").strip().lower()
    supported: TranscriptSupport = "supported" if sup == "supported" else "unsupported"
    ev = obj.get("transcript_evidence")
    ev_str = str(ev).strip() if isinstance(ev, str) and ev.strip() else None
    return AdditionEvidence(
        addition_text=addition,
        supported_by_transcript=supported,
        transcript_evidence=ev_str,
        evidence_source="gemini",
        matched_claim_text=None,
    )


def _looks_like_abbreviation_only(addition: str) -> str | None:
    """
    Detect very small additions that are just acronyms/abbreviations in parentheses,
    e.g. "(UTI)". Returns the acronym (lowercased) when detected.
    """
    s = (addition or "").strip()
    m = re.fullmatch(r"[\(\[\{]\s*([A-Za-z][A-Za-z0-9]{1,9})\s*[\)\]\}]", s)
    if not m:
        return None
    return m.group(1).lower()


def _heuristic_support_override(transcript: str, addition: str) -> tuple[TranscriptSupport, str | None] | None:
    """
    If Gemini marks an addition unsupported because it's just a standard abbreviation
    or terminology synonym, override to supported when the transcript contains the
    expanded/synonymous phrase.
    """
    t = (transcript or "").lower()
    a = (addition or "").strip()
    if not t or not a:
        return None

    ac = _looks_like_abbreviation_only(a)
    if ac and ac in _ACRONYM_EXPANSIONS:
        for exp in _ACRONYM_EXPANSIONS[ac]:
            if exp.lower() in t:
                return "supported", exp

    toks = _tokens(a)
    for term, syns in _TERM_SYNONYMS.items():
        if term in toks:
            for s in syns:
                if s.lower() in t:
                    return "supported", s
    return None


def build_addition_evidence(
    *,
    additions: Iterable[str],
    transcript: str,
    agent3_claims: list[dict[str, Any]] | None,
    allow_gemini_fallback: bool,
    gemini_model: str | None = None,
) -> list[AdditionEvidence]:
    out: list[AdditionEvidence] = []
    claims = agent3_claims or []

    for add in additions:
        addition = str(add or "").strip()
        if not addition:
            continue

        match = _best_claim_match(addition, claims) if claims else None
        if match is not None:
            out.append(_evidence_from_claim(addition, match))
            continue

        if allow_gemini_fallback:
            try:
                ev = _gemini_check(transcript, addition, model=gemini_model)
                if ev.supported_by_transcript == "unsupported":
                    override = _heuristic_support_override(transcript, addition)
                    if override is not None:
                        supported, evidence = override
                        ev.supported_by_transcript = supported
                        if evidence and not ev.transcript_evidence:
                            ev.transcript_evidence = evidence
                out.append(ev)
            except Exception:
                # If Gemini is unavailable, still apply our lightweight
                # abbreviation/synonym overrides before falling back to unknown.
                supported_override = _heuristic_support_override(transcript, addition)
                if supported_override is not None:
                    supported, evidence = supported_override
                    out.append(
                        AdditionEvidence(
                            addition_text=addition,
                            supported_by_transcript=supported,
                            transcript_evidence=evidence,
                            evidence_source="none",
                            matched_claim_text=None,
                        )
                    )
                    continue
                out.append(
                    AdditionEvidence(
                        addition_text=addition,
                        supported_by_transcript="unknown",
                        transcript_evidence=None,
                        evidence_source="none",
                        matched_claim_text=None,
                    )
                )
        else:
            out.append(
                AdditionEvidence(
                    addition_text=addition,
                    supported_by_transcript="unknown",
                    transcript_evidence=None,
                    evidence_source="none",
                    matched_claim_text=None,
                )
            )

    return out

