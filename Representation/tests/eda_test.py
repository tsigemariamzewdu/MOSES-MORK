import unittest

from Representation.eda import apply_deduction
from Representation.factor_graph import FactorGraph, SubtreeVariable, PairwiseFactor


class TestEdaDeduction(unittest.TestCase):
    def test_apply_deduction_ignores_self_loop_factors(self):
        fg = FactorGraph()
        fg.add_variable(SubtreeVariable("A"))
        fg.add_variable(SubtreeVariable("B"))

        fg.add_factor(PairwiseFactor("A", "A", (0.9, 0.7)))
        fg.add_factor(PairwiseFactor("A", "B", (0.8, 0.6)))

        apply_deduction(fg)

        self.assertEqual(len(fg.factors), 2)
        self.assertIsNotNone(fg.get_factor(("A", "A")))
        self.assertIsNotNone(fg.get_factor(("A", "B")))

    def test_apply_deduction_creates_missing_transitive_factor(self):
        fg = FactorGraph()
        fg.add_variable(SubtreeVariable("A", (0.6, 0.7)))
        fg.add_variable(SubtreeVariable("B", (0.7, 0.8)))
        fg.add_variable(SubtreeVariable("C", (0.4, 0.6)))

        fg.add_factor(PairwiseFactor("A", "B", (0.8, 0.9)))
        fg.add_factor(PairwiseFactor("B", "C", (0.6, 0.85)))

        apply_deduction(fg)

        inferred = fg.get_factor(("A", "C"))
        self.assertIsNotNone(inferred)
        self.assertTrue(inferred.inferred)


if __name__ == "__main__":
    unittest.main()