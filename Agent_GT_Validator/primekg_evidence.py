from __future__ import annotations

import os
import re
import threading
from typing import Any

from .schemas import AdditionEvidence, TranscriptSupport

# One shared explorer for parallel GT-validator workers (avoids N× full CSV loads).
_primekg_explorer_lock = threading.Lock()
_primekg_explorer = None
_primekg_explorer_init_failed = False
# Serialize read-only graph walks (pandas indexing is not guaranteed thread-safe).
_primekg_query_lock = threading.Lock()


def _get_shared_primekg_explorer():
    global _primekg_explorer, _primekg_explorer_init_failed
    with _primekg_explorer_lock:
        if _primekg_explorer_init_failed:
            return None
        if _primekg_explorer is None:
            try:
                from Agent_3.MKG import PrimeKGExplorer

                _primekg_explorer = PrimeKGExplorer()
            except Exception:
                _primekg_explorer_init_failed = True
                return None
        return _primekg_explorer


def _env_int(name: str, default: int) -> int:
    v = (os.getenv(name) or "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = (os.getenv(name) or "").strip()
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def build_primekg_context_lines(
    generated_soap: str,
    parsed_output: dict[str, Any] | None = None,
) -> list[str]:
    """
    Build the same style of PrimeKG triple lines used for Agent 3 (see pipeline.agent_interface).

    Returns deduplicated lines like ``- x — relation — y`` (no leading bullet in returned strings;
    each line is the raw triple text without the "- " prefix used in prompts — caller may prefix).
    """
    parsed_output = parsed_output or {}
    try:
        from Agent_3.MKG import extract_medical_terms
    except Exception:
        return []

    hops = _env_int("MKG_KG_HOPS", 1)
    max_pool = _env_int("MKG_KG_MAX_ROWS", 2000)
    max_gliner_terms = _env_int("MKG_GLINER_MAX_TERMS", 12)
    max_query_terms = _env_int("MKG_PRIMEKG_QUERY_TERMS", 8)
    # Cap triple lines fed into prompts (matches pipeline.agent_interface._primekg_context default).
    max_rows = _env_int("MKG_KG_CONTEXT_MAX_LINES", 40)
    gliner_threshold = _env_float("MKG_GLINER_THRESHOLD", 0.35)

    candidates: list[str] = []
    try:
        for t in extract_medical_terms(
            generated_soap,
            threshold=gliner_threshold,
        )[:max_gliner_terms]:
            if len(t) >= 2:
                candidates.append(t)
    except Exception:
        candidates = []

    if not candidates:
        for k in ("diagnosis", "primary_diagnosis", "condition", "disease", "problem"):
            v = parsed_output.get(k)
            if isinstance(v, str) and v.strip() and len(v.strip()) >= 2:
                candidates.append(v.strip())
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and item.strip() and len(item.strip()) >= 2:
                        candidates.append(item.strip())
        if not candidates:
            snippet = (generated_soap or "").strip()[:2000]
            if snippet:
                candidates = [snippet]

    if not candidates:
        return []

    explorer = _get_shared_primekg_explorer()
    if explorer is None:
        return []

    out_lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for c in candidates[:max_query_terms]:
        try:
            with _primekg_query_lock:
                df = explorer.get_related_triples(c, hops=hops, max_rows=max_pool)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            x = str(row.get("x_name", "")).strip()
            rel = str(row.get("relation", "")).strip()
            y = str(row.get("y_name", "")).strip()
            if not (x and rel and y):
                continue
            key = (x, rel, y)
            if key in seen:
                continue
            seen.add(key)
            out_lines.append(f"{x} — {rel} — {y}")
            if len(out_lines) >= max_rows:
                break
        if len(out_lines) >= max_rows:
            break

    return out_lines


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\\-]{2,}", re.IGNORECASE)

_STOP = {
    "the",
    "and",
    "for",
    "with",
    "patient",
    "history",
    "reports",
    "denies",
    "plan",
    "assessment",
    "subjective",
    "objective",
    "mild",
    "moderate",
    "severe",
}


def _informative_tokens(addition: str) -> list[str]:
    s = (addition or "").lower()
    toks = _TOKEN_RE.findall(s)
    out: list[str] = []
    seen: set[str] = set()
    for t in toks:
        if t in _STOP or len(t) < 3:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _heuristic_primekg_support(
    addition: str,
    kg_lines: list[str],
    *,
    hit_ratio_supported: float | None = None,
) -> tuple[TranscriptSupport, str | None]:
    """
    Best-effort: addition is supported by PrimeKG context if a sufficient fraction of
    informative tokens from the addition appear in at least one triple line.
    """
    ratio = hit_ratio_supported
    if ratio is None:
        ratio = float(os.getenv("GTV_PRIMEKG_HIT_RATIO", "0.34"))

    lines = [ln.strip() for ln in kg_lines if ln.strip()]
    if not lines:
        return "unknown", None

    blob = " ".join(lines).lower()
    add_toks = _informative_tokens(addition)
    if not add_toks:
        return "unknown", None

    best_line: str | None = None
    best_score = 0.0
    for line in lines:
        lt = line.lower()
        hits = sum(1 for t in add_toks if t in lt)
        score = hits / len(add_toks)
        if score > best_score:
            best_score = score
            best_line = line

    if best_score >= ratio:
        return "supported", (best_line or "")[:800]

    overall_hits = sum(1 for t in add_toks if t in blob)
    if overall_hits == 0:
        return "unsupported", None

    return "unknown", None


def _claim_map(claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    m: dict[str, dict[str, Any]] = {}
    for cl in claims:
        ct = str(cl.get("claim_text") or "").strip()
        if ct and ct not in m:
            m[ct] = cl
    return m


def enrich_addition_evidence_primekg(
    items: list[AdditionEvidence],
    *,
    generated_soap: str,
    agent3_claims: list[dict[str, Any]] | None,
    parsed_output: dict[str, Any] | None = None,
    enabled: bool,
) -> list[AdditionEvidence]:
    """
    Attach ``supported_by_primekg`` / ``primekg_evidence`` without changing transcript fields.

    Priority:
    1) If the evidence row was matched to an Agent 3 claim with non-empty ``medical_knowledge_evidence``,
       treat as PrimeKG-supported using that string (aligns with pipeline MKG channel).
    2) Otherwise, heuristic overlap between addition text and triple lines from PrimeKG.
    """
    if not enabled or not items:
        return items

    kg_lines = build_primekg_context_lines(generated_soap, parsed_output)
    claims = agent3_claims if isinstance(agent3_claims, list) else []
    by_text = _claim_map(claims)

    out: list[AdditionEvidence] = []
    for ev in items:
        pk_sup: TranscriptSupport = "unknown"
        pk_ev: str | None = None

        mct = (ev.matched_claim_text or "").strip()
        if mct and mct in by_text:
            cl = by_text[mct]
            mke = cl.get("medical_knowledge_evidence")
            if isinstance(mke, str) and mke.strip():
                pk_sup = "supported"
                pk_ev = mke.strip()[:800]

        if pk_sup != "supported":
            h, line = _heuristic_primekg_support(ev.addition_text, kg_lines)
            pk_sup = h
            if line:
                pk_ev = line

        out.append(ev.model_copy(update={"supported_by_primekg": pk_sup, "primekg_evidence": pk_ev}))
    return out
