import unittest

from Soap_Agents.Agent_GT_Validator.addition_evidence import build_addition_evidence


class TestAdditionEvidence(unittest.TestCase):
    def test_maps_to_supported_claim(self):
        additions = ["Blood pressure is 128/82 mmHg."]
        transcript = "Doctor: Your blood pressure is 128/82 mmHg."
        agent3_claims = [
            {
                "claim_text": "Blood pressure: 128/82 mmHg",
                "support_status": "supported",
                "transcript_evidence": "Your blood pressure is 128/82 mmHg",
            }
        ]
        out = build_addition_evidence(
            additions=additions,
            transcript=transcript,
            agent3_claims=agent3_claims,
            allow_gemini_fallback=False,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].supported_by_transcript, "supported")
        self.assertEqual(out[0].evidence_source, "agent3_claims")
        self.assertIn("blood pressure", (out[0].transcript_evidence or "").lower())

    def test_unknown_when_no_match_and_no_fallback(self):
        out = build_addition_evidence(
            additions=["Some unmatched statement"],
            transcript="",
            agent3_claims=[],
            allow_gemini_fallback=False,
        )
        self.assertEqual(out[0].supported_by_transcript, "unknown")
        self.assertEqual(out[0].evidence_source, "none")

    def test_abbreviation_only_can_be_supported_by_expansion(self):
        # Gemini fallback is disabled here; we simulate the scenario by ensuring
        # no agent3 claim matches, so the entry becomes unknown without fallback.
        # The override is only applied on gemini results, so this test uses fallback
        # but with a transcript that clearly supports the expansion.
        out = build_addition_evidence(
            additions=["(UTI)", "hematuria"],
            transcript="Doctor: You might have a urinary tract infection. Also watch for blood in urine.",
            agent3_claims=[],
            allow_gemini_fallback=True,
            gemini_model=None,
        )
        # We don't assert evidence_source (could be gemini), but we do require
        # that these are not marked unsupported purely due to wording.
        self.assertEqual(out[0].addition_text, "(UTI)")
        self.assertEqual(out[0].supported_by_transcript, "supported")
        self.assertEqual(out[1].addition_text, "hematuria")
        self.assertEqual(out[1].supported_by_transcript, "supported")


if __name__ == "__main__":
    unittest.main()

