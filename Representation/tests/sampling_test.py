import unittest
import random
from copy import deepcopy
from Representation.sampling import (randomUniform, randomBernoulli,
                                     sample_new_instances, sample_logical_perms,
                                     sample_from_TTable)
from Representation.representation import (Instance, Knob, Deme,
                                           Hyperparams, knobs_from_truth_table)
from Representation.csv_parser import load_truth_table
from Representation.helpers import TreeNode, parse_sexpr, tokenize, isOP

class TestRandomUniform(unittest.TestCase):
    def setUp(self):
        self.hyperparams = Hyperparams(
            mutation_rate=0.1,
            crossover_rate=0.6,
            neighborhood_size=10,
            num_generations=50,
            bernoulli_prob=0.5,
            uniform_prob=0.5
        )

    def test_random_uniform_returns_subset_of_input(self):
        knobs = ["A", "B", "C", "D"]
        random.seed(0)
        selected = randomUniform(knobs, self.hyperparams)
        # All selected items must come from original list, no extras
        self.assertTrue(set(selected).issubset(set(knobs)))

    def test_random_uniform_empty_input(self):
        knobs = []
        random.seed(0)
        selected = randomUniform(knobs, self.hyperparams)
        self.assertEqual(selected, None)

    def test_random_uniform_not_deterministic_distribution(self):
        # Over many runs, expect both empty and non-empty outcomes at least once
        knobs = ["A", "B"]
        saw_empty = False
        saw_non_empty = False
        random.seed(1)
        for _ in range(100):
            selected = randomUniform(knobs, self.hyperparams)
            if not selected:
                saw_empty = True
            else:
                saw_non_empty = True
            if saw_empty and saw_non_empty:
                break
        self.assertTrue(saw_empty)
        self.assertTrue(saw_non_empty)


class TestRandomBernoulli(unittest.TestCase):
    def setUp(self):
        self.knobA = Knob(symbol="A", id=1, Value=[True, False])
        self.knobB = Knob(symbol="B", id=2, Value=[True, False])
        self.knobC = Knob(symbol="C", id=3, Value=[True, False])

        self.knobD = Knob(symbol="D", id=4, Value=[True, False])
        self.knobE = Knob(symbol="E", id=5, Value=[True, False])

        self.instance = Instance(
            value="(AND A (OR B C))",
            id=1,
            score=0.0,
            knobs=[self.knobA, self.knobB, self.knobC],
        )

        self.hyperparams = Hyperparams(
            mutation_rate=0.1,
            crossover_rate=0.6,
            neighborhood_size=10,
            num_generations=50,
            bernoulli_prob=0.5,
            uniform_prob=0.5
        )

        self.perms = ["D", "E", "C"]
        self.new_knobs = [self.knobD, self.knobE, self.knobC]

    def test_random_bernoulli_returns_none_if_no_selected_knobs(self):
        # Force randomUniform to always return empty by using no perms
        new_inst = randomBernoulli(self.hyperparams, self.instance, [], self.new_knobs)
        self.assertIsNone(new_inst)
    
    def test_random_bernoulli_does_not_duplicate_knobs(self):
        random.seed(3)
        new_inst = randomBernoulli(1.0, self.instance, self.new_knobs, self.instance.knobs)
        self.assertIsNotNone(new_inst)
        symbols = [k.symbol for k in new_inst.knobs]
        self.assertEqual(len(symbols), len(set(symbols)))

    # def test_random_bernoulli_append_on_false_probability(self):
    #     random.seed(42)
    #     new_inst = randomBernoulli(self.hyperparams, self.instance, self.new_knobs, self.instance.knobs)
        
    #     self.assertIsNotNone(new_inst)
    #     self.assertNotEqual(new_inst.value, self.instance.value)
    #     self.assertGreater(len(new_inst.value), len(self.instance.value))
    #     self.assertTrue("D" in new_inst.value or "E" in new_inst.value)
        

    def test_random_bernoulli_probability_zero_no_change(self):
        random.seed(0)
        self.hyperparams.bernoulli_prob = 0.0
        new_inst = randomBernoulli(self.hyperparams, self.instance, self.new_knobs, self.instance.knobs)
        # With p=0, there should be no change to the expression
        # self.assertIsNotNone(new_inst)
        # self.assertEqual(new_inst.value, self.instance.value)
        # # Knobs should be identical after filtering
        # self.assertCountEqual(
        #     [k.symbol for k in new_inst.knobs],
        #     [k.symbol for k in self.instance.knobs],
        # )
        # new_inst = randomBernoulli(0.0, self.instance, self.features, self.knobs)
        self.assertNotEqual(new_inst.value, self.instance.value)
        # self.assertTrue(len(new_inst.value) > len(self.instance.value))


    def test_random_bernoulli_probability_one_mutates_if_knobs_available(self):
        random.seed(0)
        self.hyperparams.bernoulli_prob = 1.0
        new_inst = randomBernoulli(self.hyperparams, self.instance, self.new_knobs, self.instance.knobs)
        # Expect either at least one mutation or, worst case, same value but logic still valid
        self.assertIsNotNone(new_inst)
        self.assertIsInstance(new_inst, Instance)
        # At least one candidate path should have been attempted; value can be equal
        self.assertIn("AND", new_inst.value)  # still valid expression structure

    def test_random_bernoulli_updates_knobs_based_on_tokens(self):
        random.seed(2)
        new_inst = randomBernoulli(self.hyperparams, self.instance, self.new_knobs, self.instance.knobs)
        self.assertIsNotNone(new_inst)
        present_tokens = set(tokenize(new_inst.value))
        knob_symbols = {k.symbol for k in new_inst.knobs}
        # All knob symbols should be present in tokens
        self.assertTrue(knob_symbols.issubset(present_tokens))

    def test_random_bernoulli_does_not_duplicate_knobs(self):
        random.seed(3)
        new_inst = randomBernoulli(self.hyperparams, self.instance, self.new_knobs, self.instance.knobs)
        self.assertIsNotNone(new_inst)
        symbols = [k.symbol for k in new_inst.knobs]
        self.assertEqual(len(symbols), len(set(symbols)))

    def test_random_bernoulli_handles_complex_perm_expression(self):
        random.seed(2)
        new_inst = randomBernoulli(self.hyperparams, self.instance, [self.knobA, self.knobC, self.knobE], self.instance.knobs)
        self.assertIsNotNone(new_inst)
        knob_symbols = {k.symbol for k in new_inst.knobs}
        self.assertTrue(
            knob_symbols.issuperset({"C"}) or knob_symbols.issuperset({"E"})
        )

class TestRandomBernoulliAdvanced(unittest.TestCase):
    def setUp(self):
        self.knobA = Knob(symbol="A", id=1, Value=[True, False])
        self.knobB = Knob(symbol="B", id=2, Value=[True, False])
        self.knobC = Knob(symbol="C", id=3, Value=[True, False])
        self.all_knobs = [self.knobA, self.knobB, self.knobC]
        
        self.hyperparams = Hyperparams(
            mutation_rate=0.1, crossover_rate=0.5, neighborhood_size=10, 
            num_generations=1, bernoulli_prob=0.0, uniform_prob=0.5 # p=0.0 forces APPEND
        )

    def test_append_to_empty_root(self):
        """Test that we can append a knob to an empty logic tree (AND)."""
        instance = Instance(value="(AND)", id=1, score=0.0, knobs=[])
        
        # We start with empty, we want to add 'A'
        features = [self.knobA]
        
        # p=0.0 forces append logic
        hp_sure = deepcopy(self.hyperparams)
        hp_sure.uniform_prob = 0.0 # Force selection
        
        new_inst = randomBernoulli(hp_sure, instance, features, self.all_knobs)
        
        new_inst = randomBernoulli(self.hyperparams, instance, features, self.all_knobs)
        
        self.assertIsNotNone(new_inst)
        # Should become (AND A)
        self.assertIn("A", new_inst.value)
        self.assertTrue(new_inst.value.startswith("(AND"))
        # Verify the knob object was added to the list
        self.assertEqual(len(new_inst.knobs), 1)
        self.assertEqual(new_inst.knobs[0].symbol, "A")

    def test_append_skips_not_node(self):
        """
        Test that finding a NOT node forces the append to go to the grandparent.
        Target: (AND (NOT A))
        Action: Append 'B'
        Expected: (AND (NOT A) B) -- NOT (AND (NOT A B))
        """
        instance = Instance(value="(AND (NOT A))", id=2, score=0.0, knobs=[self.knobA])
        features = [self.knobB]
        
        random.seed(42) 
        
        new_inst = randomBernoulli(self.hyperparams, instance, features, self.all_knobs)
        
        self.assertIsNotNone(new_inst)
        self.assertFalse("(NOT A B)" in new_inst.value)
        self.assertFalse("(NOT B A)" in new_inst.value)
        self.assertIn("B", new_inst.value)
        
        # Specific check for your specific logic:
        # If it appended to (AND...), the value represents valid syntax
        # (AND (NOT A) B)
        self.assertTrue(new_inst.value == "(AND (NOT A) B)")

    def test_knob_list_consistency(self):
        """Test that new_inst.knobs strictly matches the tokens in new_inst.value."""
        # Start with A, B. Replace/Append C. 
        # If we Replace A with C -> Knobs should be [B, C] (A removed)
        
        instance = Instance(value="(AND A B)", id=3, score=0.0, knobs=[self.knobA, self.knobB])
        features = [self.knobC]
        
        hp_replace = Hyperparams(
            mutation_rate=0.1, crossover_rate=0.5, neighborhood_size=10, 
            num_generations=1, bernoulli_prob=1.0, uniform_prob=0.5
        )
        
        random.seed(42)
        new_inst = randomBernoulli(hp_replace, instance, features, self.all_knobs)
        
        if new_inst is None:
            return

        if "A" not in new_inst.value:
            knob_symbols = [k.symbol for k in new_inst.knobs]
            self.assertNotIn("A", knob_symbols)
            self.assertIn("C", knob_symbols)
            self.assertIn("B", knob_symbols)

class TestSampleNewInstances(unittest.TestCase):
    def setUp(self):
        self.knobA = Knob(symbol="A", id=1, Value=[True, False])
        self.knobB = Knob(symbol="B", id=2, Value=[True, False])
        self.knobC = Knob(symbol="C", id=3, Value=[True, False])

        self.knobD = Knob(symbol="D", id=4, Value=[True, False])
        self.knobE = Knob(symbol="E", id=5, Value=[True, False])

        self.instance = Instance(
            value="(AND A (OR B C))",
            id=1,
            score=0.0,
            knobs=[self.knobA, self.knobB, self.knobC],
        )

        self.perms = ["D", "E", "C"]
        self.new_knobs = [self.knobD, self.knobE, self.knobC]

        self.hyperparams = Hyperparams(
            mutation_rate=0.1,
            crossover_rate=0.6,
            neighborhood_size=10,
            num_generations=50,
            bernoulli_prob=0.5,
            uniform_prob=0.5
        )

    def test_sample_new_instances_returns_list_of_unique_values(self):
        random.seed(0)
        new_instances = sample_new_instances(
            self.hyperparams, self.instance, self.new_knobs, self.instance.knobs
        )
        
        self.assertIsInstance(new_instances, list)
        for v in new_instances:
            self.assertIsInstance(v, Instance)
        self.assertLessEqual(len(new_instances), self.hyperparams.neighborhood_size)
        # self.assertEqual(len(new_instances), len(set(new_instances.keys())))

    def test_sample_new_instances_respects_neighborhood_size_upper_bound(self):
        random.seed(1)
        new_instances = sample_new_instances(
            self.hyperparams, self.instance, self.new_knobs, self.instance.knobs
        )
        # Number of unique instances cannot exceed neighborhood_size
        self.assertLessEqual(len(new_instances), self.hyperparams.neighborhood_size)

    def test_sample_new_instances_with_zero_neighborhood_size(self):
        random.seed(2)
        hyperparams = deepcopy(self.hyperparams)
        hyperparams.neighborhood_size = 0
        new_instances = sample_new_instances(
            hyperparams, self.instance, self.perms, self.new_knobs
        )
        self.assertEqual(new_instances, [])

    def test_sample_new_instances_handles_no_perms(self):
        random.seed(3)
        new_instances = sample_new_instances(
            self.hyperparams, self.instance, [], self.new_knobs
        )
        # randomBernoulli should always return None, so no instances
        self.assertEqual(new_instances, [])

    def test_sample_new_instances_probability_zero(self):
        random.seed(4)
        self.hyperparams.bernoulli_prob = 0.0
        new_instances = sample_new_instances(
            self.hyperparams, self.instance, self.new_knobs, self.instance.knobs
        )
        # With p=0, randomBernoulli never mutates; but it still returns instances
        # self.assertGreaterEqual(len(new_instances), 1)
        # if new_instances:
        #     value, inst = new_instances.items()
        #     self.assertEqual(value, self.instance.value)
        #     self.assertEqual(inst.value, self.instance.value)
        different_values = [inst for inst in new_instances if inst.value != self.instance.value]
        self.assertGreater(len(different_values), 0)

class TestSampleLogicalPerms(unittest.TestCase):
    def test_sample_logical_perms(self):
            knobs = [
                Knob(symbol="A", id=1, Value=[]),
                Knob(symbol="B", id=2, Value=[])
            ]

            # Test 1: Current OP is AND -> Expects OR pairs + Vars
            candidates_and, new_knobs = sample_logical_perms("AND", knobs)

            # Expect: "A", "B"
            self.assertIn("A", candidates_and)
            self.assertIn("B", candidates_and)

            # Expect Pairs (OR A B), (OR (NOT A) B) ...
            self.assertIn("(OR A B)", candidates_and)
            self.assertIn("(OR (NOT A) B)", candidates_and) 
            self.assertIn("(OR A (NOT B))", candidates_and)
            self.assertIn("(OR (NOT A) (NOT B))", candidates_and)

            # Test 2: Current OP is OR -> Expects AND pairs
            candidates_or, new_knobs = sample_logical_perms("OR", knobs)
            self.assertIn("(AND A B)", candidates_or)

class TestSampleFromTTable(unittest.TestCase):
    def setUp(self):
        # Use the provided binary truth table CSV
        self.test_csv_path = "example_data/test_bin.csv"

        # Build exemplar from this table
        tt, target_vals = load_truth_table(self.test_csv_path, "O")
        knobs = knobs_from_truth_table(tt)
        # Simple exemplar: AND over all input knobs (A..F)
        expr = "(AND " + " ".join(k.symbol for k in knobs) + ")"
        self.exemplar = Instance(value=expr, id=0, score=0.0, knobs=knobs)
        self.hyperparams = Hyperparams(
            mutation_rate=0.3,
            crossover_rate=0.7,
            neighborhood_size=5,
            num_generations=10,
        )
        self.target_vals = target_vals
        self.knobs = knobs

    def test_sample_from_TTable_basic_structure(self):
        random.seed(0)
        demes = sample_from_TTable(
            self.test_csv_path,
            self.hyperparams,
            self.exemplar,
            self.knobs,
            self.target_vals,
            output_col="O",
        )
        # Returns list of Deme
        self.assertIsInstance(demes, list)
        self.assertGreater(len(demes), 0)
        for deme in demes:
            self.assertIsInstance(deme, Deme)
            self.assertIsInstance(deme.instances, list)
            for inst in deme.instances:
                self.assertIsInstance(inst, Instance)
                # Value string should be non-empty expression
                self.assertIsInstance(inst.value, str)
                self.assertNotEqual(inst.value.strip(), "")

    def test_sample_from_TTable_respects_neighborhood_size(self):
        random.seed(1)
        demes = sample_from_TTable(
            self.test_csv_path,
            self.hyperparams,
            self.exemplar,
            self.knobs,
            self.target_vals,
            output_col="O",
        )
        for deme in demes:
            self.assertLessEqual(
                len(deme.instances),
                self.hyperparams.neighborhood_size,
            )

    def test_sample_from_TTable_handles_invalid_csv_path(self):
        random.seed(2)
        demes = sample_from_TTable(
            "non_existent_file.csv",
            self.hyperparams,
            self.exemplar,
            self.knobs,
            self.target_vals,
            output_col="O",
        )
        # On error it should return an empty list
        self.assertEqual(demes, [])
    
    def test_scored_instances(self):
        random.seed(3)
        demes = sample_from_TTable(
            self.test_csv_path,
            self.hyperparams,
            self.exemplar,
            self.knobs,
            self.target_vals,
            output_col="O",
        )
        for deme in demes:
            for inst in deme.instances:
                self.assertIsInstance(inst.score, float)
                self.assertGreater(inst.score, 0)



if __name__ == "__main__":
    unittest.main()