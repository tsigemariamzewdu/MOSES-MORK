import math
import csv
from typing import List, Dict, Tuple

def calculate_entropy(values: List[bool]) -> float:
    """
    Calculates the Shannon entropy of a list of boolean values.
    H(S) = -p1 log2(p1) - p2 log2(p2)
    """
    if not values:
        return 0.0
    
    total = len(values)
    true_count = sum(1 for v in values if v)
    false_count = total - true_count
    
    if true_count == 0 or false_count == 0:
        return 0.0
    
    p_true = true_count / total
    p_false = false_count / total

    H_s = -p_true * math.log2(p_true) - p_false * math.log2(p_false)

    return H_s

def calculate_information_gain(target: List[bool], feature: List[bool]) -> float:
    """
    Calculates Information Gain (IG) of a feature with respect to a target.
    IG(Target, Feature) = H(Target) - H(Target | Feature)
    """
    if len(target) != len(feature):
        raise ValueError("Target and Feature lists must have the same length")
    
    # H(Target)
    entropy_target = calculate_entropy(target)
    
    # Splits based on feature
    total = len(feature)
    
    feature_true_indices = [i for i, x in enumerate(feature) if x]
    feature_false_indices = [i for i, x in enumerate(feature) if not x]
    
    p_feature_true = len(feature_true_indices) / total
    p_feature_false = len(feature_false_indices) / total
    
    target_given_true = [target[i] for i in feature_true_indices]
    target_given_false = [target[i] for i in feature_false_indices]


    
    entropy_given_true = calculate_entropy(target_given_true)
    entropy_given_false = calculate_entropy(target_given_false)
    
    conditional_entropy = (p_feature_true * entropy_given_true) + (p_feature_false * entropy_given_false)
    
    return entropy_target - conditional_entropy

def select_features(csv_path: str, target_col: str, k: int = None, threshold: float = 0.0) -> List[Tuple[str, float]]:
    """
    Selects top features from a CSV file based on Information Gain.
    
    Args:
        csv_path: Path to the CSV file.
        target_col: Name of the output/target column.
        k: Number of top features to return. If None, returns all satisfying threshold.
        threshold: Minimum Information Gain required.
        
    Returns:
        List of tuples (feature_name, information_gain_score), sorted by score descending.
    """
    # 1. Load Data
    columns: Dict[str, List[bool]] = {}
    target_values: List[bool] = []
    
    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
                
            # Initialize lists for all headers
            for field in reader.fieldnames:
                if field != target_col:
                    columns[field] = []
            
            for row in reader:
                try:
                    # Parse Target
                    if target_col in row:
                        val = row[target_col].strip().upper()
                        target_values.append(val in ('1', 'TRUE', 'T', 'YES'))
                    else:
                        raise ValueError(f"Target column '{target_col}' not found in row")
                    
                    # Parse Features
                    for field in columns.keys():
                        val = row[field].strip().upper()
                        columns[field].append(val in ('1', 'TRUE', 'T', 'YES'))
                        
                except Exception as row_error:
                    continue # Skip malformed rows
                    
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return []

    if not target_values:
        print("Error: No target values found.")
        return []

    # 2. Calculate IG for each feature
    feature_scores = []
    
    for feature_name, feature_vals in columns.items():
        if len(feature_vals) != len(target_values):
            # Skipping if lengths mismatch 
            continue
            
        ig = calculate_information_gain(target_values, feature_vals)
        if ig >= threshold:
            feature_scores.append((feature_name, ig))
            
    # 3. Sort and Filter
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    if k is not None:
        return feature_scores[:k]
        
    return feature_scores



# if __name__ == "__main__":
#     # Read test_bin.csv and run feature selection
#     csv_file = "example_data/test_bin.csv"
#     target_column = "O"
    
#     print(f"Running feature selection on {csv_file}")
#     print(f"Target column: {target_column}\n")
    
#     # Get all features ranked by information gain
#     results = select_features(csv_file, target_column)
    
#     print("Feature Selection Results:")
#     print("-" * 50)
#     for feature_name, ig_score in results:
#         print(f"{feature_name}: {ig_score:.4f}")
    
#     print("\n" + "=" * 50)
#     print(f"\nTop 3 features:")
#     top_3 = select_features(csv_file, target_column, k=3)
#     for feature_name, ig_score in top_3:
#         print(f"  {feature_name}: {ig_score:.4f}")
    
#     print(f"\nFeatures with IG > 0.1:")
#     filtered = select_features(csv_file, target_column, threshold=0.1)
#     for feature_name, ig_score in filtered:
#         print(f"  {feature_name}: {ig_score:.4f}")
