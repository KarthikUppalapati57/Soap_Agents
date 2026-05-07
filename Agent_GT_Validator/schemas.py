from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SoapSection = Literal["Subjective", "Objective", "Assessment", "Plan", "Other"]


class ExtractedEntities(BaseModel):
    """NER-like extracted surface forms with light categorization."""

    all_entities: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    labs: list[str] = Field(default_factory=list)


class RecallScore(BaseModel):
    """Set-based recall on extracted entities: |GT ∩ GEN| / |GT|."""

    gt_count: int
    gen_count: int
    overlap_count: int
    recall: float
    gt_only: list[str] = Field(default_factory=list)
    gen_only: list[str] = Field(default_factory=list)
    overlap: list[str] = Field(default_factory=list)


class SectionAlignment(BaseModel):
    section: SoapSection
    gt_present: bool
    gen_present: bool
    token_f1: Optional[float] = None
    rouge_l_f1: Optional[float] = None


class StructuralAlignmentScore(BaseModel):
    overall_token_f1: Optional[float] = None
    overall_rouge_l_f1: Optional[float] = None
    by_section: list[SectionAlignment] = Field(default_factory=list)
    assessment_alignment_score: Optional[float] = None
    plan_alignment_score: Optional[float] = None


class JudgeSectionGrade(BaseModel):
    section: SoapSection
    grade: Literal["A", "B", "C", "D", "F"]
    summary: str
    omissions: list[str] = Field(default_factory=list)
    additions: list[str] = Field(default_factory=list)


class JudgeSectionDiscrepancy(BaseModel):
    """Per-section omissions/additions before transcript support is applied."""

    section: SoapSection
    omissions: list[str] = Field(default_factory=list)
    additions: list[str] = Field(default_factory=list)


class ExpertJudgeDiscrepancies(BaseModel):
    """Step-1 judge output: lists only (no letter grades)."""

    model: str
    omissions: list[str] = Field(default_factory=list)
    additions: list[str] = Field(default_factory=list)
    hallucinations_or_unjustified_inferences: list[str] = Field(default_factory=list)
    section_discrepancies: list[JudgeSectionDiscrepancy] = Field(default_factory=list)
    rubric_notes: Optional[str] = None


class JudgeSectionGradeOutcome(BaseModel):
    section: SoapSection
    grade: Literal["A", "B", "C", "D", "F"]
    summary: str


class ExpertJudgeGradingResult(BaseModel):
    """Step-2 judge output: grades derived from omissions + unsupported additions only."""

    model: str
    overall_grade: Literal["A", "B", "C", "D", "F"]
    overall_summary: str
    section_grades: list[JudgeSectionGradeOutcome] = Field(default_factory=list)
    rubric_notes: Optional[str] = None


class ExpertJudgeReport(BaseModel):
    model: str
    overall_grade: Literal["A", "B", "C", "D", "F"]
    overall_summary: str
    omissions: list[str] = Field(default_factory=list)
    additions: list[str] = Field(default_factory=list)
    hallucinations_or_unjustified_inferences: list[str] = Field(default_factory=list)
    section_grades: list[JudgeSectionGrade] = Field(default_factory=list)
    rubric_notes: Optional[str] = None
    addition_evidence: list["AdditionEvidence"] = Field(default_factory=list)


TranscriptSupport = Literal["supported", "unsupported", "unknown"]


class AdditionEvidence(BaseModel):
    addition_text: str
    supported_by_transcript: TranscriptSupport
    transcript_evidence: Optional[str] = None
    evidence_source: Literal["agent3_claims", "gemini", "none"] = "none"
    matched_claim_text: Optional[str] = None


class GTValidatorReport(BaseModel):
    item: str
    output_dir: str

    ground_truth_soap: str
    generated_soap: str

    entities_gt: ExtractedEntities
    entities_generated: ExtractedEntities

    clinical_recall_all: RecallScore
    clinical_recall_symptoms: RecallScore
    clinical_recall_medications: RecallScore
    clinical_recall_labs: RecallScore

    structural_alignment: StructuralAlignmentScore

    expert_judge: Optional[ExpertJudgeReport] = None
    expert_judge_error: Optional[str] = None
    addition_evidence: list[AdditionEvidence] = Field(default_factory=list)


ExpertJudgeReport.model_rebuild()

