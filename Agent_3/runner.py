"""Run the claim-verification agent once and return structured output."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from google.genai import Client, types

from .claim_agent import build_prompt, model_name
from .env_setup import load_env
from .schemas import BenchmarkScores, ClaimVerificationInput, ClaimVerificationResult




def _max_tokens() -> int | None:
    """Resolve MAX_TOKENS from project .env files first, then os.environ.

    `load_dotenv(override=False)` does not override variables already set in the
    shell, so a stale exported MAX_TOKENS can silently ignore .env (e.g. 2048 in
    the shell vs 4096 in .env). Reading the file directly avoids that.
    """
    here = Path(__file__).resolve().parent
    for env_path in (here.parent / ".env", here / ".env"):
        if not env_path.is_file():
            continue
        raw = dotenv_values(env_path).get("MAX_TOKENS")
        if raw is None or str(raw).strip() == "":
            continue
        try:
            n = int(str(raw).strip().strip('"').strip("'"))
        except ValueError:
            continue
        if n > 0:
            return n

    v = (os.getenv("MAX_TOKENS") or "").strip()
    if not v:
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    return n if n > 0 else None


def _clean_json_text(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned


def verify_claims(payload: ClaimVerificationInput, client: Client) -> ClaimVerificationResult:
    load_env()
    prompt = build_prompt(payload)

    max_out = _max_tokens()
    resp = client.models.generate_content(
        model=model_name(),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=max_out,
        ),
    )

    final_text = (resp.text or "").strip()
    cleaned = _clean_json_text(final_text)

    try:
        return ClaimVerificationResult.model_validate_json(cleaned)
    except Exception:
        # Always save the output so it can be inspected when validation fails.
        try:
            with open("final_text.txt", "w", encoding="utf-8") as f:
                f.write(final_text)
        except Exception:
            pass
        raise


def benchmark_from_row(row: dict) -> dict:
    """Build a dict suitable for BenchmarkScores from a v2_results row."""
    return {k: row.get(k) for k in BenchmarkScores.model_fields}
