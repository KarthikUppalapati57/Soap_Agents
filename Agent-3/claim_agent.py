"""ADK root agent: transcript-grounded SOAP claim verification."""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent
from google.genai import types

from schemas import ClaimVerificationInput, ClaimVerificationResult

_DEFAULT_MODEL = os.getenv("AGENT3_GEMINI_MODEL", "gemini-2.5-flash")

_INSTRUCTION = """You are a clinical documentation auditor.

You receive one JSON object (matching the input schema) with:
1) transcript — raw doctor-patient conversation (sole source of truth for support)
2) generated_soap — the SOAP note to audit
3) benchmark — numeric/text scores from an automated evaluator (ADVISORY ONLY)

Hard rules:
- Use ONLY the transcript, generated_soap, and benchmark fields. No outside medical knowledge or web search.
- For each support decision, the transcript is authoritative. If benchmark scores or raw_output disagree with the transcript, follow the transcript.
- Extract claims from generated_soap only (not from the transcript as separate claims).

Claim definition:
- One claim = one checkable clinical/factual statement (symptom, vital, exam finding, diagnosis wording, medication, plan step, attribution of speech, etc.).
- Split compound sentences into multiple claims when they assert multiple facts.
- Do not merge unrelated facts.

support_status (exactly one of: "supported" | "unsupported"):
- supported: the claim is clearly stated or unambiguously implied in the transcript.
- unsupported: absent, contradicted, or relies on clinical inference not explicitly grounded in what was said in the transcript.

For every unsupported claim, rationale must say what is missing or contradictory in the transcript.
For supported claims, set transcript_evidence to a brief verbatim snippet or tight paraphrase from the transcript when possible.

Optional: if benchmark.accuracy_total_claims is set, your number of claims may differ from it; if so, briefly explain in benchmark_reconciliation_note. Never change a claim's support_status to match the benchmark.

Respond ONLY with JSON matching the output schema."""

root_agent = LlmAgent(
    name="soap_claim_verifier",
    model=_DEFAULT_MODEL,
    description=(
        "Extracts factual claims from a SOAP note and labels each supported or "
        "unsupported against the raw transcript."
    ),
    instruction=_INSTRUCTION,
    input_schema=ClaimVerificationInput,
    output_schema=ClaimVerificationResult,
    output_key="claim_verification",
    include_contents="none",
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
)
