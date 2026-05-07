from __future__ import annotations

import os
import re

from .schemas import ExtractedEntities

_gliner_model = None
_gliner_model_id: str | None = None

_DEFAULT_MODEL_ID = "urchade/gliner_mediumv2.1"

# Keep this aligned with the clinical categories you asked for.
_SYMPTOM_LABELS: tuple[str, ...] = ("symptom", "sign")
_MED_LABELS: tuple[str, ...] = ("medication", "drug")
_LAB_LABELS: tuple[str, ...] = ("lab test", "diagnostic test", "vital sign")

# A broader set for "all entities" recall.
_ALL_LABELS: tuple[str, ...] = (
    "disease",
    "disorder",
    "medication",
    "drug",
    "symptom",
    "sign",
    "procedure",
    "treatment",
    "diagnostic test",
    "lab test",
    "vital sign",
    "anatomical structure",
    "allergy",
)


def _get_gliner(model_id: str | None = None):
    global _gliner_model, _gliner_model_id
    mid = model_id or os.environ.get("GTV_GLINER_MODEL", _DEFAULT_MODEL_ID)
    if _gliner_model is None or _gliner_model_id != mid:
        try:
            from gliner import GLiNER
        except ModuleNotFoundError:
            return None

        _gliner_model = GLiNER.from_pretrained(mid)
        _gliner_model_id = mid
    return _gliner_model


def _chunk_text(text: str, *, max_chars: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for p in parts:
        if not p:
            continue
        sep = 1 if cur else 0
        if n + len(p) + sep > max_chars and cur:
            chunks.append(" ".join(cur))
            cur = [p]
            n = len(p)
        else:
            cur.append(p)
            n += len(p) + sep
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _normalize_entity(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_entities(
    text: str,
    *,
    labels: tuple[str, ...],
    threshold: float,
    model_id: str | None,
    max_chunk_chars: int,
) -> list[str]:
    model = _get_gliner(model_id)
    if model is None:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in _chunk_text(text, max_chars=max_chunk_chars):
        ents = model.predict_entities(chunk, list(labels), threshold=threshold)
        for e in ents:
            t = _normalize_entity(e.get("text") or "")
            if not t:
                continue
            if t not in seen:
                seen.add(t)
                ordered.append(t)
    return ordered


_FALLBACK_TOKEN_RE = re.compile(r"[a-z][a-z0-9\\-]{2,}", re.IGNORECASE)
_FALLBACK_STOP = {
    "the",
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
    "reports",
    "denies",
    "history",
    "present",
    "presents",
    "plan",
    "subjective",
    "objective",
    "assessment",
}

_FALLBACK_SYMPTOMS = {
    "pain",
    "fever",
    "nausea",
    "vomiting",
    "diarrhea",
    "cough",
    "fatigue",
    "headache",
    "swelling",
    "stiffness",
    "tenderness",
    "insomnia",
    "dizziness",
    "sob",
    "dyspnea",
    "palpitations",
}

_FALLBACK_LABS = {
    "xray",
    "x-ray",
    "mri",
    "ct",
    "cbc",
    "bmp",
    "cmp",
    "lipid",
    "profile",
    "blood",
    "pressure",
    "bp",
    "heart",
    "rate",
    "temperature",
    "respiratory",
    "rr",
}


def _fallback_entities(text: str) -> ExtractedEntities:
    toks = [_normalize_entity(t) for t in _FALLBACK_TOKEN_RE.findall(text or "")]
    toks = [t for t in toks if t and t not in _FALLBACK_STOP]
    seen: set[str] = set()
    ordered: list[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    symptoms = [t for t in ordered if t in _FALLBACK_SYMPTOMS]
    labs = [t for t in ordered if t in _FALLBACK_LABS]
    meds = [
        t
        for t in ordered
        if t.endswith(("pril", "olol", "sartan", "statin", "prazole", "caine", "cillin"))
        or t in {"acetaminophen", "ibuprofen", "naproxen", "aspirin", "metformin"}
    ]
    return ExtractedEntities(
        all_entities=ordered,
        symptoms=symptoms,
        medications=meds,
        labs=labs,
    )


def extract_entities_for_validation(
    soap_text: str,
    *,
    threshold: float | None = None,
    model_id: str | None = None,
    max_chunk_chars: int | None = None,
) -> ExtractedEntities:
    """
    Extract entity surface forms from a SOAP note for overlap-based scoring.

    Defaults can be tuned via env:
    - GTV_GLINER_THRESHOLD (default 0.35)
    - GTV_GLINER_MODEL
    - GTV_GLINER_CHUNK_CHARS (default 2500)
    """
    thr = threshold if threshold is not None else float(os.getenv("GTV_GLINER_THRESHOLD", "0.35"))
    chunk_chars = max_chunk_chars if max_chunk_chars is not None else int(
        os.getenv("GTV_GLINER_CHUNK_CHARS", "2500")
    )

    # If GLiNER isn't installed in the current environment, fall back to a
    # simple token/lexicon extractor so the validator still runs end-to-end.
    if _get_gliner(model_id) is None:
        return _fallback_entities(soap_text)

    all_ents = _extract_entities(
        soap_text,
        labels=_ALL_LABELS,
        threshold=thr,
        model_id=model_id,
        max_chunk_chars=chunk_chars,
    )
    symptoms = _extract_entities(
        soap_text,
        labels=_SYMPTOM_LABELS,
        threshold=thr,
        model_id=model_id,
        max_chunk_chars=chunk_chars,
    )
    meds = _extract_entities(
        soap_text,
        labels=_MED_LABELS,
        threshold=thr,
        model_id=model_id,
        max_chunk_chars=chunk_chars,
    )
    labs = _extract_entities(
        soap_text,
        labels=_LAB_LABELS,
        threshold=thr,
        model_id=model_id,
        max_chunk_chars=chunk_chars,
    )
    return ExtractedEntities(
        all_entities=all_ents,
        symptoms=symptoms,
        medications=meds,
        labs=labs,
    )

