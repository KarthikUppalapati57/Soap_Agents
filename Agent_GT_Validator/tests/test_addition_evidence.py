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


if __name__ == "__main__":
    unittest.main()

