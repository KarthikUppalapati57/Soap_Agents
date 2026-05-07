from __future__ import annotations

import re
from typing import Mapping

from .schemas import SoapSection

_SECTION_ORDER: tuple[SoapSection, ...] = (
    "Subjective",
    "Objective",
    "Assessment",
    "Plan",
)


def _normalize_heading(s: str) -> str:
    return re.sub(r"[\s:]+", "", (s or "").strip().lower())


_HEADING_TO_SECTION: dict[str, SoapSection] = {
    "subjective": "Subjective",
    "objective": "Objective",
    "assessment": "Assessment",
    "plan": "Plan",
}


def parse_soap_sections(text: str) -> dict[SoapSection, str]:
    """
    Best-effort parsing of a SOAP note into canonical sections.

    Input is typically free-form text containing headings like:
    - "Subjective:" / "Objective:" / "Assessment:" / "Plan:"
    Headings are matched case-insensitively and tolerate extra whitespace.
    """
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return {s: "" for s in _SECTION_ORDER}

    # Identify heading lines and their positions.
    heading_re = re.compile(
        r"(?im)^(?P<h>(subjective|objective|assessment|plan))\s*:?\s*$"
    )

    matches = list(heading_re.finditer(raw))
    if not matches:
        # Fallback: try inline headings "Subjective:" within lines.
        inline_re = re.compile(r"(?im)^(?P<h>(subjective|objective|assessment|plan))\s*:\s*")
        matches = list(inline_re.finditer(raw))
        if not matches:
            # No headings detected; treat as unstructured and return empty sections.
            return {s: "" for s in _SECTION_ORDER}

    spans: list[tuple[SoapSection, int, int]] = []
    for i, m in enumerate(matches):
        h = _normalize_heading(m.group("h"))
        sec = _HEADING_TO_SECTION.get(h)
        if not sec:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        spans.append((sec, start, end))

    out: dict[SoapSection, str] = {s: "" for s in _SECTION_ORDER}
    covered = [False] * len(raw)
    for sec, start, end in spans:
        chunk = raw[start:end].strip()
        if out[sec]:
            out[sec] = (out[sec].rstrip() + "\n\n" + chunk).strip()
        else:
            out[sec] = chunk
        for j in range(max(0, start), min(len(raw), end)):
            covered[j] = True

    return out


def section_present_map(sections: Mapping[SoapSection, str]) -> dict[SoapSection, bool]:
    return {s: bool((sections.get(s) or "").strip()) for s in _SECTION_ORDER}

