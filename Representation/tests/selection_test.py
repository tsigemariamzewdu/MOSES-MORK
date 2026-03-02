import unittest
from Representation.selection import (
    select_top_k,
    tournament_selection,
    boltzmann_roulette_selection,
)
from Representation.representation import (
    Instance,
    Deme,
    Knob,
    Hyperparams
)

class TestSelection(unittest.TestCase):
    def setUp(self):
        self.ITable = [
            {"A": True,  "B": True,  "O": True},
            {"A": True,  "B": False, "O": False},
            {"A": False, "B": True,  "O": False},
            {"A": False, "B": False, "O": False},
        ]
        self.sketch = "(AND $ $)"

    def test_select_top_k(self):
            
            kA = Knob(symbol="A", id=1, Value=[True, False])
            kB = Knob(symbol="B", id=2, Value=[True, False])
            kC = Knob(symbol="C", id=3, Value=[True, False])
            
            i1 = Instance(value="(AND A)", id=1, score=0.9, knobs=[kA])
            i2 = Instance(value="(AND A B)", id=2, score=0.6, knobs=[kA, kB])
            i3 = Instance(value="(AND A B C)", id=3, score=0.4, knobs=[kA, kB, kC])
            deme = Deme([i1, i2, i3], "Deme-01", Hyperparams(0.1, 0.1, 10, 5))
            
            top = select_top_k(deme, k=2)
            self.assertEqual(len(top), 2)
            self.assertEqual(top[0].id, 1)
            self.assertEqual(top[1].id, 2)
    
    def test_tournament_selection(self):
        kA = Knob(symbol="A", id=1, Value=[True, False])
        kB = Knob(symbol="B", id=2, Value=[True, False])
        kC = Knob(symbol="C", id=3, Value=[True, False])
        
        i1 = Instance(value="(AND A)", id=1, score=0.9, knobs=[kA])
        i2 = Instance(value="(AND A B)", id=2, score=0.6, knobs=[kA, kB])
        i3 = Instance(value="(AND A B C)", id=3, score=0.4, knobs=[kA, kB, kC])
        deme = Deme([i1, i2, i3], "Deme-01", Hyperparams(0.1, 0.1, 10, 5))

        selected = tournament_selection(deme, k=2, tournament_size=2)
        self.assertEqual(len(selected), 2)

        winners = tournament_selection(deme, k=1, tournament_size=3)
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0].id, 1)

    def test_boltzmann_selection_includes_elite(self):
        random_seed = 42
        import random
        random.seed(random_seed)

        kA = Knob(symbol="A", id=1, Value=[True, False])
        kB = Knob(symbol="B", id=2, Value=[True, False])
        kC = Knob(symbol="C", id=3, Value=[True, False])

        i1 = Instance(value="(AND A)", id=1, score=0.9, knobs=[kA])
        i2 = Instance(value="(AND A B)", id=2, score=0.6, knobs=[kA, kB])
        i3 = Instance(value="(AND A B C)", id=3, score=0.4, knobs=[kA, kB, kC])
        deme = Deme([i1, i2, i3], "Deme-01", Hyperparams(0.1, 0.1, 10, 5))

        selected = boltzmann_roulette_selection(deme, k=2, temperature=1.0, elite_count=1)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].id, 1)

    def test_boltzmann_selection_zero_temp_falls_back_to_greedy(self):
        kA = Knob(symbol="A", id=1, Value=[True, False])
        kB = Knob(symbol="B", id=2, Value=[True, False])
        kC = Knob(symbol="C", id=3, Value=[True, False])

        i1 = Instance(value="(AND A)", id=1, score=0.9, knobs=[kA])
        i2 = Instance(value="(AND A B)", id=2, score=0.6, knobs=[kA, kB])
        i3 = Instance(value="(AND A B C)", id=3, score=0.4, knobs=[kA, kB, kC])
        deme = Deme([i1, i2, i3], "Deme-01", Hyperparams(0.1, 0.1, 10, 5))

        selected = boltzmann_roulette_selection(deme, k=2, temperature=0.0, elite_count=0)
        self.assertEqual([x.id for x in selected], [1, 2])

    def test_boltzmann_selection_caps_k_to_population(self):
        kA = Knob(symbol="A", id=1, Value=[True, False])
        i1 = Instance(value="(AND A)", id=1, score=0.9, knobs=[kA])
        deme = Deme([i1], "Deme-01", Hyperparams(0.1, 0.1, 10, 5))

        selected = boltzmann_roulette_selection(deme, k=5, temperature=1.0, elite_count=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].id, 1)