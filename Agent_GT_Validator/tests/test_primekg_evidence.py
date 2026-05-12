import unittest

from Soap_Agents.Agent_GT_Validator.primekg_evidence import (
    _heuristic_primekg_support,
    enrich_addition_evidence_primekg,
)
from Soap_Agents.Agent_GT_Validator.schemas import AdditionEvidence


class TestPrimeKGEvidence(unittest.TestCase):
    def test_heuristic_supported_when_tokens_hit_triple(self):
        lines = ["osteoarthritis — indication — knee pain"]
        st, ev = _heuristic_primekg_support(
            "Mild osteoarthritis of the knee on exam",
            lines,
            hit_ratio_supported=0.25,
        )
        self.assertEqual(st, "supported")
        self.assertIsNotNone(ev)

    def test_heuristic_unsupported_no_overlap(self):
        lines = ["diabetes — drug — metformin"]
        st, _ = _heuristic_primekg_support("Isolated acute appendicitis", lines)
        self.assertEqual(st, "unsupported")

    def test_enrich_from_claim_mkg_string(self):
        ev = AdditionEvidence(
            addition_text="extra",
            supported_by_transcript="unsupported",
            evidence_source="agent3_claims",
            matched_claim_text="claim one",
        )
        claims = [{"claim_text": "claim one", "medical_knowledge_evidence": "X — rel — Y"}]
        out = enrich_addition_evidence_primekg(
            [ev],
            generated_soap="",
            agent3_claims=claims,
            parsed_output=None,
            enabled=True,
        )
        self.assertEqual(out[0].supported_by_primekg, "supported")
        self.assertIn("rel", out[0].primekg_evidence or "")


if __name__ == "__main__":
    unittest.main()
