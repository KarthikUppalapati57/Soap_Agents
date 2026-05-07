import unittest

from Soap_Agents.Agent_GT_Validator.section_parser import parse_soap_sections


class TestSectionParser(unittest.TestCase):
    def test_basic_headings(self):
        text = (
            "Subjective:\nA\n\n"
            "Objective:\nB\n\n"
            "Assessment:\nC\n\n"
            "Plan:\nD\n"
        )
        sec = parse_soap_sections(text)
        self.assertEqual(sec["Subjective"], "A")
        self.assertEqual(sec["Objective"], "B")
        self.assertEqual(sec["Assessment"], "C")
        self.assertEqual(sec["Plan"], "D")

    def test_no_headings(self):
        text = "freeform\nnote"
        sec = parse_soap_sections(text)
        self.assertEqual(sec["Other"], "freeform\nnote")

    def test_inline_heading(self):
        text = "Subjective: A\nObjective: B\n"
        sec = parse_soap_sections(text)
        self.assertIn("A", sec["Subjective"])
        self.assertIn("B", sec["Objective"])


if __name__ == "__main__":
    unittest.main()

