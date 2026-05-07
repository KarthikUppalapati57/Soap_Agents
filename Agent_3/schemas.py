from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class BenchmarkScores(BaseModel):
    """Subset of Agent-2 v2_results row used as advisory context only."""

    id: Optional[int] = None
    accuracy_total_claims: Optional[int] = None
    accuracy_correct: Optional[int] = None
    accuracy_incorrect: Optional[int] = None
    accuracy_score: Optional[float] = None
    accuracy_reason: Optional[str] = None
    completeness_total_details: Optional[int] = None
    completeness_captured: Optional[int] = None
    completeness_missing: Optional[int] = None
    completeness_score: Optional[float] = None
    completeness_reason: Optional[str] = None
    communication_total_issues: Optional[int] = None
    communication_quality_score: Optional[float] = None
    communication_quality_reason: Optional[str] = None
    context_total_details: Optional[int] = None
    context_captured: Optional[int] = None
    context_missing: Optional[int] = None
    context_awareness_score: Optional[float] = None
    context_awareness_reason: Optional[str] = None
    instruction_total_sections: Optional[int] = None
    instruction_complete_sections: Optional[int] = None
    instruction_following_score: Optional[float] = None
    instruction_following_reason: Optional[str] = None
    overall_score: Optional[float] = None
    status: Optional[str] = None
    raw_output: Optional[str] = None


class ClaimVerificationInput(BaseModel):
    transcript: str = Field(description="Raw doctor-patient transcript.")
    generated_soap: str = Field(description="Generated SOAP note text.")
    benchmark: BenchmarkScores = Field(
        description="Prior automated benchmark scores; advisory only."
    )
    medical_knowledge_terms: str = Field(description="List of MKG terms.")


SoapSection = Literal["Subjective", "Objective", "Assessment", "Plan", "Other"]
SupportStatus = Literal["supported", "unsupported"]


class Claim(BaseModel):
    claim_text: str = Field(description="One atomic factual/clinical statement from the SOAP.")
    soap_section: SoapSection
    support_status: SupportStatus
    rationale: str = Field(
        description="Why this is supported or unsupported, grounded in the transcript and/or medical_knowledge_terms.",
    )
    transcript_evidence: Optional[str] = Field(
        default=None,
        description="Short quote or tight paraphrase from the transcript when the transcript supports the claim; null otherwise.",
    )
    medical_knowledge_evidence: Optional[str] = Field(
        default=None,
        description="Short quote or tight paraphrase from medical_knowledge_terms when the MKG supports the claim; null otherwise.",
    )
    medical_knowledge_rationale: Optional[str] = Field(
        default=None,
        description="How medical_knowledge_terms supports or fails to support the claim; null if MKG is not part of the basis.",
    )


class ClaimVerificationResult(BaseModel):
    claims: list[Claim] = Field(description="All extracted claims from the SOAP note.")
    benchmark_reconciliation_note: Optional[str] = Field(
        default=None,
        description="Optional note if claim counts diverge from benchmark; do not adjust claims to match benchmark.",
    )
