import unittest

from Soap_Agents.Agent_GT_Validator.metrics import entity_recall, token_f1, rouge_l_f1


class TestMetrics(unittest.TestCase):
    def test_entity_recall(self):
        gt = ["a", "b", "c"]
        gen = ["b", "c", "d"]
        r = entity_recall(gt, gen)
        self.assertEqual(r.gt_count, 3)
        self.assertEqual(r.gen_count, 3)
        self.assertEqual(r.overlap_count, 2)
        self.assertAlmostEqual(r.recall, 2 / 3, places=6)

    def test_token_f1(self):
        self.assertAlmostEqual(token_f1("a b c", "a b"), 0.8, places=6)

    def test_rouge_l(self):
        self.assertGreaterEqual(rouge_l_f1("a b c", "a b"), 0.0)


if __name__ == "__main__":
    unittest.main()

