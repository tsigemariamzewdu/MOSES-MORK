import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Representation.helpers import TreeNode, parse_sexpr, tokenize, isOP
from Representation.representation import (Instance, Knob,
                                           Deme, Hyperparams,
                                           FitnessOracle,
                                           knobs_from_truth_table)
from Representation.csv_parser import load_truth_table

from Feature_selection_algo.interaction_mrmr import interaction_aware_mrmr, feature_order
from reduct.enf.main import reduce
from hyperon import MeTTa
import csv
from typing import List, Dict
from copy import deepcopy
import random

def sample_logical_perms(current_op: str, variables: List[Knob]) -> List[str]:
    """
    Generates a 'menu' of new Boolean logic pieces (proposals).
    """
    if current_op not in ["AND", "OR"] or not variables:
        return None, None
    
    candidates = []
    candidates.extend([v.symbol for v in variables])

    # If current is AND, make OR pairs. If OR, make AND pairs.
    pair_op = "OR" if current_op == "AND" else "AND"

    var_symbols = [v.symbol for v in variables]

    for i in range(len(var_symbols)):
        for j in range(i + 1, len(var_symbols)):
            s1 = var_symbols[i]
            s2 = var_symbols[j]
            # Form pairs including negations
            candidates.append(f"({pair_op} {s1} {s2})")
            candidates.append(f"({pair_op} (NOT {s1}) {s2})")
            candidates.append(f"({pair_op} {s1} (NOT {s2}))")
            candidates.append(f"({pair_op} (NOT {s1}) (NOT {s2}))")
    
    return candidates, variables

def randomUniform(knobs, hyperparams: Hyperparams):
    """
    Perform uniform random sampling to select knobs.
    """
    if not knobs:
        return
    
    selected_knobs = []
    for knob in knobs:
        if random.random() > hyperparams.uniform_prob:
            selected_knobs.append(knob)

    return selected_knobs   

def randomBernoulli(hyperparams: Hyperparams, instance: Instance, features: List[Knob], knobs: List[Knob]) -> Instance:
    """
    Perform Bernoulli sampling to select a knob for replacement.
    
    Args:
        p (float): Probability of selecting a knob.
        hyperparams: neihborhood_size identification.
        instance: The instance object (exemplar).
        knobs (List): List of knob objects to sample from.
        
    Returns:
        A set of newly generated instances.
    """
    instanceExp = deepcopy(instance.value)
    sexp = tokenize(instanceExp)
    op = sexp[1] if sexp else None
    root = parse_sexpr(sexp)
    perms, new_knobs = sample_logical_perms(op, features)
    selected_knobs = randomUniform(perms, hyperparams)



    if not selected_knobs:
        return

    candidates = [[]]
    queue = [(root, [])]
    mutant_root = deepcopy(root)


    while queue:
        curr_node, curr_path = queue.pop(0)
        
        for i, child in enumerate(curr_node.children):
            child_path = curr_path + [i]
            candidates.append(child_path)
            
            if not child.is_leaf():
                queue.append((child, child_path))

    new_inst = Instance(
                value=str(mutant_root),
                id=instance.id,
                score=0.0,
                knobs=deepcopy(instance.knobs)
            )
    
    for path in candidates:

        if not selected_knobs:
            break
        # symbol = selected_knobs.pop(0)
        symbol = selected_knobs[0]

        # Navigate to the PARENT of the target node using the path
        parent = mutant_root
        valid_path = True
        for idx in path[:-1]:
            if idx < len(parent.children):
                parent = parent.children[idx]
            else:
                valid_path = False
                break
        
        target_idx = path[-1] if path else -1
        if not valid_path:
            continue  # Skip this mutation as the target no longer exists
        
        tokens = tokenize(symbol)
        if len(tokens) > 1 and parent.label == "OR" and tokens[1] == "OR":
            tokens[tokens.index("OR")] = "AND"
            symbol = " ".join(tokens).replace("( ", "(").replace(" )", ")")

        if len(tokens) > 1 and parent.label == "AND" and tokens[1] == "AND":
            tokens[tokens.index("AND")] = "OR"
            symbol = " ".join(tokens).replace("( ", "(").replace(" )", ")")


        rand = random.random()
        # print("Random: ", rand, " Bernoulli Prob: ", hyperparams.bernoulli_prob)
        if rand > hyperparams.bernoulli_prob:
            append_target = parent
            if append_target.label == "NOT":
                if len(path) < 2:
                    continue
                
                grandparent = mutant_root
                gp_valid = True
                for idx in path[:-2]:
                    if idx < len(grandparent.children):
                        grandparent = grandparent.children[idx]
                    else:
                        gp_valid = False
                        break
                
                if not gp_valid: continue
                append_target = grandparent

            if append_target.label not in ["AND", "OR"]:
                continue

            is_duplicate = False
            for child in append_target.children:
                if str(child) == symbol:
                    is_duplicate = True; break
            
            if not is_duplicate:
                append_target.children.append(TreeNode(symbol))
                selected_knobs.pop(0)
                
        mutant_value = str(mutant_root)
        if mutant_value == instanceExp:
            continue

        new_inst.value = mutant_value
        for t in tokens:
            if isOP(t) or t in ['(', ')']:
                continue

            knob = next((k for k in knobs if k.symbol == t), None)
            if knob and knob.symbol not in [k.symbol for k in new_inst.knobs]:
                new_inst.knobs.append(knob)
            
            new_knob = next((k for k in new_knobs if k.symbol == t), None)
            if new_knob and new_knob.symbol not in [k.symbol for k in new_inst.knobs]:
                new_inst.knobs.append(new_knob)

    present_tokens = set(tokenize(new_inst.value))
    new_inst.knobs = [k for k in new_inst.knobs if k.symbol in present_tokens]

    return new_inst


def sample_new_instances(hyperparams: Hyperparams, instance: Instance, features: List, knobs: List[Knob]) -> List[Instance]:
    """
    Sample new instances using Bernoulli sampling.
    
    Args:
        p (float): Probability of selecting a knob.
        hyperparams: hyperparameters for sampling.
        instance: The instance object (exemplar).
        knobs (List): List of knob objects to sample from.
        
    Returns:
        A dict of newly generated instances.
    """
    new_instances = {}
    id_counter = 1
    for _ in range(hyperparams.neighborhood_size):
        new_inst = randomBernoulli(hyperparams, instance, features, knobs)
        if new_inst and new_inst.value not in new_instances:
            new_inst.id = instance.id + id_counter
            new_instances[new_inst.value] = new_inst
            id_counter += 1
    
    return list(new_instances.values())

def sample_from_deme(deme: Deme, hyperparams: Hyperparams, exemplar: Instance, global_knobs: List[Knob],
                     features: List[Knob], csv_path: str, fitness: FitnessOracle, metta: MeTTa) -> Deme:
    """
    sample_from_deme: Samples new instances for a given deme using features 
        extracted from a truth table CSV file.
    Returns: A new deme with the sampled instances added.
    """
    if features:
        features = extract_features(csv_path, output_col='O')
        selected_features = random.sample(features, 1)
        selected_knobs = [k for k in global_knobs if k.symbol in selected_features]
    else: selected_knobs = features
    # print("-"*100)
    # print(selected_features)
    new_instances = sample_new_instances(hyperparams, exemplar, selected_knobs, global_knobs)
    new_instances = reduce_and_score(new_instances, fitness, metta)
    deme.instances.extend(new_instances)

    return deme

def extract_features(csv_path: str, output_col: str = 'O'):
    """
    Extracts features from a truth table CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing the truth table.
        output_col (str): Name of the output/target column in the CSV.
        
    Returns:
        A list of features.
    """
    order = feature_order(csv_path, output_col)
    features = interaction_aware_mrmr(
        csv_path=csv_path,
        target_col=output_col,
        k=5,
        max_interaction_order=order,
        output_type='subsets'
    )
    print("Features: ", features)
    return features

def reduce_and_score(instances: List[Instance], fitness: FitnessOracle, metta: MeTTa) -> List[Instance]:
    """
    Reduces the instances using MeTTa and scores them using the fitness oracle.
    
    Args:
        instances (List[Instance]): List of instances to reduce and score.
        fitness (FitnessOracle): The fitness oracle to evaluate the instances.
        metta (MeTTa): The MeTTa interpreter for reduction.
        
    Returns:
        A list of reduced and scored instances.
    """
    unique_instances = {}
    for inst in instances:
        reduced = reduce(metta, inst.value)
        if isinstance(reduced, list) and len(reduced) > 0:
            inst.value = str(reduced[0])
        else:
            inst.value = str(reduced)

        present_tokens = set(tokenize(inst.value))
        inst.knobs = [k for k in inst.knobs if k.symbol in present_tokens]

        inst.score = fitness.get_fitness(inst)
        if inst.value not in unique_instances:
            unique_instances[inst.value] = inst
            
    return list(unique_instances.values())

def sample_from_TTable(csv_path: str, hyperparams: Hyperparams, exemplar: Instance, knobs: List[Knob], target_vals: List[bool] ,output_col: str = 'O'):
    """
    Samples demes from a truth table CSV file using interaction-aware mRMR feature selection.
    Args:
        csv_path (str): Path to the CSV file containing the truth table.
        hyperparams (Hyperparams): Hyperparameters for sampling.
        exemplar (Instance): The exemplar instance to base sampling on.
        output_col (str): Name of the output/target column in the CSV.
    Returns:
        List[Deme]: A list of sampled demes.
    """
    features = extract_features(csv_path, output_col)

    demes = []
    metta = MeTTa()
    fitness = FitnessOracle(target_vals)

    for feat in features:
        selected_features = [k for k in knobs if k.symbol in (feat if isinstance(feat, (list, tuple)) else [feat])]
        instances = sample_new_instances(hyperparams, exemplar, selected_features, exemplar.knobs)
        unique_instances = reduce_and_score(instances, fitness, metta)
        # for inst in instances.values():
        #     reduced = reduce(metta, inst.value)
        #     # inst.value = reduced
        #     if isinstance(reduced, list) and len(reduced) > 0:
        #         inst.value = str(reduced[0])
        #     else:
        #         inst.value = str(reduced)


        #     present_tokens = set(tokenize(inst.value))
        #     inst.knobs = [k for k in inst.knobs if k.symbol in present_tokens]

        #     inst.score = fitness.get_fitness(inst)
        #     if inst.value not in unique_instances:
        #         unique_instances[inst.value] = inst
            
        demes.append(Deme(instances=list(unique_instances), id=(len(demes)), q_hyper=hyperparams))
        
    return demes