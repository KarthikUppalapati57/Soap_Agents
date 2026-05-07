import json
import os
from pathlib import Path

import dotenv
from google import genai
from google.genai import types

dotenv.load_dotenv()

client = genai.Client(api_key=os.getenv("gemini_api_key"))

model = "gemini-3-pro-preview"

def _max_tokens() -> int | None:
    v = (os.getenv("MAX_TOKENS") or "").strip()
    if not v:
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    return n if n > 0 else None


def build_evaluator_prompt(generated_soap: str, ground_truth: str) -> str:
    """Build the evaluator system/user prompt from the generated SOAP and ground-truth SOAP only."""
    return f"""
# Role

You are a senior clinical documentation evaluator and quality assurance specialist. Your job is to evaluate a generated SOAP note in light of a reference ground-truth SOAP note. You are objective, conservative, and prioritize patient safety and factual accuracy.

# Situation

You are given exactly two documents:

1. **Generated SOAP note** — the note to evaluate
2. **Ground-truth SOAP note** — a reference for what good documentation of the same encounter might look like

The generated SOAP note does NOT need to match the ground truth verbatim. Your task is to evaluate whether the generated SOAP note is **clinically accurate, complete, safe, and well-structured**, using the ground truth as a **reference**, not as a strict answer key. You do not receive raw unstructured clinical data; base your judgment only on these two SOAP documents.

# Task

Evaluate the generated SOAP note using the following rubric:

## Evaluation Criteria

### 1. Clinical Accuracy

* Does the generated note align with the ground truth on key facts (symptoms, vitals, findings) where they should agree?
* Are symptoms and findings interpreted in a way that is consistent with the ground truth?
* Are vitals and measurements plausible and consistent with the ground truth when comparable?
* Penalize incorrect medical information or clear contradictions with the ground truth when both notes address the same facts

### 2. Completeness

* Compared to the ground-truth SOAP, are major symptoms and clinical details reasonably covered in the generated note?
* Are important items from the ground truth missing from the generated note without good reason?
* Are medications, allergies, or vitals omitted when they appear in the ground truth?

### 3. Hallucination / Fabrication

* Did the generated note invent symptoms, diagnoses, medications, or tests that are not supported by the ground-truth encounter (or that contradict it)?
* Penalize unsupported or contradictory information relative to the ground-truth SOAP

### 4. Clinical Reasoning (Assessment Quality)

* Does the assessment logically follow subjective + objective?
* Is reasoning medically sound?
* Penalize unsupported diagnoses

### 5. Safety

* Are recommendations safe?
* Are urgent conditions handled appropriately?
* Penalize unsafe or risky advice

### 6. SOAP Format Quality

* Proper Subjective section
* Proper Objective section
* Proper Assessment section
* Proper Plan section
* Clear formatting

# Scoring

Score each category from 1–5:

1 = Poor
2 = Weak
3 = Acceptable
4 = Good
5 = Excellent

# Pass Criteria

A note should PASS if:

* Overall score ≥ 4.0
* Safety ≥ 4
* Hallucination ≥ 4

Otherwise FAIL.

# Additional Instructions

* Be strict but fair
* Do not penalize wording differences
* Focus on meaning and clinical validity
* If uncertain, choose the more conservative score
* Explain reasoning briefly

# Output Schema

Return ONLY valid JSON in this format:

{{
"scores": {{
"clinical_accuracy": number,
"completeness": number,
"hallucination": number,
"clinical_reasoning": number,
"safety": number,
"format": number
}},
"section_scores": {{
"subjective": number,
"objective": number,
"assessment": number,
"plan": number
}},
"overall_score": number,
"pass": boolean,
"major_issues": [
"string"
],
"minor_issues": [
"string"
],
"reasoning": "string"
}}

# Inputs

Generated SOAP:
{generated_soap}

Ground-truth SOAP:
{ground_truth}

"""

def evaluate_soap(generated_soap: str, ground_truth: str) -> dict:
    """Evaluate the generated SOAP note in light of the ground-truth SOAP note."""
    prompt = build_evaluator_prompt(generated_soap, ground_truth)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    max_output_tokens=_max_tokens(),
                )
    )
    return json.loads(response.text.strip().removeprefix("```json").removesuffix("```"))

_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    generated_soap = (_DIR / "generated_soap.txt").read_text(encoding="utf-8")
    ground_truth = (_DIR / "ground_truth.txt").read_text(encoding="utf-8")
    output = evaluate_soap(generated_soap, ground_truth)
    with open(_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2)