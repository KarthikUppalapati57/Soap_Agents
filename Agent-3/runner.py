"""Run the claim-verification agent once and return structured output."""

from __future__ import annotations

import uuid
from typing import Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from claim_agent import root_agent
from env_setup import load_env
from schemas import BenchmarkScores, ClaimVerificationInput, ClaimVerificationResult

APP_NAME = "soap_agent_3"
USER_ID = "batch_user"


def verify_claims(payload: ClaimVerificationInput) -> ClaimVerificationResult:
    load_env()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
        auto_create_session=True,
    )
    session_id = f"inv-{uuid.uuid4().hex}"
    text = payload.model_dump_json()
    user_content = types.Content(role="user", parts=[types.Part(text=text)])

    final_text: Optional[str] = None
    for event in runner.run(
        user_id=USER_ID,
        session_id=session_id,
        new_message=user_content,
    ):
        if (
            event.is_final_response()
            and event.content
            and event.content.parts
            and event.content.parts[0].text
        ):
            final_text = event.content.parts[0].text

    if not final_text:
        raise RuntimeError("Agent returned no final text response.")

    cleaned = final_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()

    return ClaimVerificationResult.model_validate_json(cleaned)


def benchmark_from_row(row: dict) -> dict:
    """Build a dict suitable for BenchmarkScores from a v2_results row."""
    return {k: row.get(k) for k in BenchmarkScores.model_fields}
