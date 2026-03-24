import random
import unittest

from main import run_moses
from Representation.csv_parser import load_truth_table
from Representation.helpers import tokenize
from Representation.representation import FitnessOracle, Hyperparams, Instance, knobs_from_truth_table


class TestMosesParity4NoOutputLeak(unittest.TestCase):
    CSV_PATH = "example_data/test_parity_5.csv"

    def _run_mode(self, fg_type: str):
        random.seed(42)
        print(self.CSV_PATH)

        input_rows, target = load_truth_table(self.CSV_PATH, output_col="O")
        self.assertTrue(input_rows, "Expected parity_5 input rows to load")
        self.assertTrue(target, "Expected parity_5 target values to load")

        knobs = knobs_from_truth_table(input_rows, exclude="O")
        self.assertTrue(knobs, "Expected non-empty knob set")
        self.assertNotIn("O", {k.symbol for k in knobs})

        hyperparams = Hyperparams(
            mutation_rate=0.3,
            crossover_rate=0.5,
            num_generations=5,
            neighborhood_size=10,
            max_iter=50,
            fg_type=fg_type,
            bernoulli_prob=0.6,
            uniform_prob=0.8,
            initial_population_size=2,
            exemplar_selection_size=7,
            min_crossover_neighbors=5,
            evidence_propagation_steps=20,
            max_dist=20,
            feature_order=5,
        )

        fitness = FitnessOracle(target)
        exemplar = Instance(value="(AND)", id=0, score=0.0, knobs=knobs)
        exemplar.score = fitness.get_fitness(exemplar)
        metapop = [exemplar]

        final_pop = run_moses(
            exemplar=exemplar,
            fitness=fitness,
            hyperparams=hyperparams,
            knobs=knobs,
            target=target,
            csv_path=self.CSV_PATH,
            metapop=metapop,
            max_iter=50,
            fg_type=fg_type,
        )

        self.assertTrue(final_pop, f"Expected non-empty final population for mode={fg_type}")

        for inst in final_pop:
            tokens = tokenize(inst.value)
            # O should not leak in to instances
            self.assertNotIn("O", tokens, f"Found leaked token O in instance: {inst.value}")
            self.assertNotIn("o", tokens, f"Found leaked token o in instance: {inst.value}")

            if inst.knobs:
                knob_symbols = {k.symbol for k in inst.knobs}
                self.assertNotIn("O", knob_symbols, f"Found leaked knob O in instance: {inst.value}")

    def test_parity4_beta_no_output_symbol_leak(self):
        self._run_mode("beta")

    def test_parity4_alpha_no_output_symbol_leak(self):
        self._run_mode("alpha")


if __name__ == "__main__":
    unittest.main()
