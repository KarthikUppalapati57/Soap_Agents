import unittest

from Soap_Agents.Agent_GT_Validator import judge


class TestJudgePrompt(unittest.TestCase):
    def test_prompt_contains_ignore_list(self):
        p = judge._prompt("GT", "GEN", "TRANSCRIPT")
        self.assertIn("Do NOT include these as omissions", p)
        self.assertIn("patient name", p.lower())
        self.assertIn("icd-10", p.lower())
        self.assertIn("Transcript:", p)

    def test_grade_rubric_uses_transcript_support(self):
        g = judge._prompt_grading("GT", "GEN", "TR", "{}", "[]")
        self.assertIn("supported_by_transcript", g)
        self.assertIn("unsupported", g.lower())
        self.assertIn("hallucinations_or_unjustified_inferences", g)


if __name__ == "__main__":
    unittest.main()

