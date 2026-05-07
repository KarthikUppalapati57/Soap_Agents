from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from google.genai import Client, types


def _default_model() -> str:
    # Keep consistent with the rest of the repo defaults.
    return os.getenv("AGENT1_GEMINI_MODEL", "gemini-2.5-flash")


def _max_tokens() -> int | None:
    v = (os.getenv("MAX_TOKENS") or "").strip()
    if not v:
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    return n if n > 0 else None


def _render_prompt_optimizations(prompt_optimizations_list: list[Any] | None) -> str:
    items = prompt_optimizations_list or []
    if not items:
        return "None."
    return "\n".join(f"- {x}" for x in items)

def generate_soap_v1(transcript: str, prompt_path: str, prompt_optimizations_list: list[Any] | None) -> str:
    """
    Generate a SOAP note from a doctor-patient transcript.

    API contract: called by `pipeline/agent_interface.py` as
    `generate_soap_v1(transcript, prompt_path, prompt_optimizations_list)`.
    """

    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    prompt = prompt_template.format(
        transcript=transcript,
        prompt_optimizations=_render_prompt_optimizations(prompt_optimizations_list),
    )

    client = Client()
    resp = client.models.generate_content(
        model=_default_model(),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=0.0,
            max_output_tokens=_max_tokens(),
        ),
    )

    return (resp.text or "").strip()

