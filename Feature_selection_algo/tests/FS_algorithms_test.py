import unittest
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Feature_selection_algo.IG_selection import select_features as ig_select
from Feature_selection_algo.interaction_mrmr import interaction_aware_mrmr, feature_order

class TestAllFeatureSelection(unittest.TestCase):
    def setUp(self):
        self.test_csv_path = "test_features_full.csv"
        # Create a dataset where:
        # A is strongly correlated with O (Copy of O)
        # B is weakly correlated (O inverted, maybe?)
        # C is constant/useless
        
        with open(self.test_csv_path, "w") as f:
            f.write("A,B,C,O\n")
            # A = O
            # B = O (Redundant with A)
            # C = Noise/Constant
            # O
            
            # Simple Pattern: O = A
            data = [
                # A  B  C  O
                ('0','0','0','0'),
                ('0','0','1','0'),
                ('0','1','0','0'), # B diverges here to show less relevance than A
                ('1','1','0','1'),
                ('1','1','1','1'),
                ('1','0','1','1'), # B diverges here 
            ]
            
            for row in data:
                f.write(",".join(row) + "\n")

    def tearDown(self):
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)

    def test_ig_selection(self):
        print("\nTesting IG Selection...")
        # A should have highest IG (Perfect match)
        # C should have very low IG
        scores = ig_select(self.test_csv_path, target_col="O")
        
        print(f"IG Scores: {scores}")
        
        # Check if A is the top feature
        self.assertEqual(scores[0][0], "A")
        
        # Check if score is high (near 1.0 entropy reduction, though base entropy might be < 1)
        self.assertGreater(scores[0][1], 0.5)
        
    def test_interaction_aware_mrmr(self):
        print("\nTesting Interaction mRMR...")
        # Testing with max_interaction_order=1 (singles)
        selected_singles = interaction_aware_mrmr(self.test_csv_path, target_col="O", k=2, max_interaction_order=1)
        print(f"Interaction (Order 1): {selected_singles}")
        
        # Should pick {A} first
        first_set = selected_singles[0][0]
        self.assertTrue("A" in first_set)
        
        # Testing with max_interaction_order=2 (pairs)
        # Since A is perfect, {A} should still be strong, but maybe {A, B} pair is considered?
        # The function returns subsets.
        selected_pairs = interaction_aware_mrmr(self.test_csv_path, target_col="O", k=2, max_interaction_order=2)
        print(f"Interaction (Order 2): {selected_pairs}")
        
        # Ensure we get results
        self.assertTrue(len(selected_pairs) > 0)

        # Test output_type='set' (Flattened)
        print("\nTesting Interaction mRMR (Flattened Set)...")
        flattened_set = interaction_aware_mrmr(self.test_csv_path, target_col="O", k=2, max_interaction_order=2, output_type='set')
        print(f"Flattened Set: {flattened_set}")
        self.assertIsInstance(flattened_set, set)
        self.assertTrue("A" in flattened_set)

        # Test output_type='subsets'
        print("\nTesting Interaction mRMR (Subsets)...")
        subsets_set = interaction_aware_mrmr(self.test_csv_path, target_col="O", k=2, max_interaction_order=2, output_type='subsets')
        print(f"Subsets Set: {subsets_set}")
        self.assertIsInstance(subsets_set, set)
        
        # We expect subsets containing 'A' since 'A' is the strong feature.
        # It might be 'A' (singleton) or ('A', 'C') (pair) depending on tie-breaking.
        has_A = False
        for item in subsets_set:
            if item == "A":
                has_A = True
            elif isinstance(item, tuple) and "A" in item:
                has_A = True
        
        self.assertTrue(has_A, "Result should contain A in some subset")

    def test_feature_order(self):
        print("\nTesting Feature Order...")
        order = feature_order(self.test_csv_path, target_col="O")
        print(f"Feature Order: {order}")
        self.assertEqual(order, 3)


if __name__ == '__main__':
    unittest.main()
