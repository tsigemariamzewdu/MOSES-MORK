from DependencyMiner.miner import DependencyMiner

from Representation.representation import *
from Representation.helpers import *
from Representation.csv_parser import load_truth_table
from Representation.selection import select_top_k, tournament_selection
from Representation.sampling import sample_from_TTable

from Variation_quantale.crossover import VariationQuantale, crossTopOne
from Variation_quantale.mutation import Mutation

from FactorGraph_EDA.beta_bp import BetaFactorGraph

import random

def run_variation(deme, fitness, hyperparams, target):

    bg = BetaFactorGraph()
    
    for generation in range(hyperparams.num_generations):
        print("-" * 60)
        print(f"\n--- Generation {generation + 1} ---")
        # Evaluate fitness of all instances in the deme

        selected_exemplars = select_top_k(deme, k=7)

        values = [inst.value for inst in deme.instances]
        weights = [inst.score for inst in deme.instances]
        # values = [inst.value for inst in selected_exemplars]
        # weights = [inst.score for inst in selected_exemplars]
        miner = DependencyMiner()
        # print(f"\tSelected number  of Exemplars for Dependency Mining: {len(values)}")
        miner.fit(values, weights)
        # print(f"\nMining for correlation for {len(values)} exemplars...")
        correlation = miner.get_meaningful_dependencies()
        # print(f"{'Pair' :>20} | {'Strength' :>6} | {'Confidence' :>8}")
        print("-" * 50)
        # for row in correlation:
            # print(f"{row['pair']:>20} | {row['strength']:.4f} | {row['confidence']:.4f}")
    
        for row in correlation:
            bg.add_dependency_rule(row['pair'], row['strength'], row['confidence'])

        if len(correlation) > 0:
            top_rule = correlation[0]
            parts = top_rule['pair'].split(' -- ')
            if len(parts) > 0:
                root_node = parts[0]
                print(f"\nDynamically setting prior for '{root_node}' based on rule: {top_rule['pair']}")

                bg.set_prior(
                    root_node, 
                    stv_strength=top_rule['strength'], 
                    stv_confidence=top_rule['confidence']
                )
        else:
            print("No correlations found. Setting default prior for 'A'.")
            # bg.set_prior("A", stv_strength=0.5, stv_confidence=0.1)
            continue
        
        bg.run_evidence_propagation(steps=20)

        # print("\n--- Final STV Results ---")
        # print(f"{'Variable':<10} | {'Strength':<8} | {'Confidence':<10} | {'Counts (a/b)'}")
        sorted_nodes = sorted(bg.nodes.items(), key=lambda item: item[1].confidence, reverse=True)
        for name, node in sorted_nodes:
            s = node.strength
            c = node.confidence
            # print(f"{name:<10} | {s:.4f}   | {c:.4f}     | {node.alpha:.1f}/{node.beta:.1f}")
        
        stv_dict = {}
        for name, node in bg.nodes.items():
            stv_dict[name] = (node.strength, node.confidence)
        

        # print(f"\nExtracted STV Values for Crossover: {stv_dict}")
        # print(f"\nSelected for Crossover (based on top nodes): {[inst.value for inst in selected_exemplars]}")
        raw_children = crossTopOne(selected_exemplars, stv_dict, target)
        
        children = []
        seen_child_values = set()
        
        # Optional: Pre-populate with existing deme values if you want to filter those out too
        seen_child_values = {inst.value for inst in deme.instances}

        for inst in raw_children:
            if inst.value not in seen_child_values:
                children.append(inst)
                seen_child_values.add(inst.value)
                # print(f"Child: {inst.value} | Score: {inst.score}")

        mut_parent = max(selected_exemplars, key=lambda x: x.score)
        mutation = Mutation(mut_parent, stv_dict, hyperparams)
        new_mutants = []

        # print(f"\nMutation of '{mut_parent.value}' | Score: {mut_parent.score}")
        child1 = mutation.execute_additive()
        if isinstance(child1, Instance):
            child1.score = fitness.get_fitness(child1)
            new_mutants.append(child1)
            # print(f"Additive Mutation Child: {child1.value} | Score: {child1.score}")

        child2 = mutation.execute_multiplicative()
        if isinstance(child2, Instance):
            child2.score = fitness.get_fitness(child2)
            new_mutants.append(child2)
            # print(f"Multiplicative Mutation Child: {child2.value} | Score: {child2.score}")


        children.extend(new_mutants)
        unique_children = []
        # Create a set of existing values in the current deme to avoid re-adding them
        existing_values = {inst.value for inst in deme.instances}
        
        for child in children:
            if child.value not in existing_values:
                unique_children.append(child)
                existing_values.add(child.value) # Add to set to prevent duplicates within the batch too
            else:
                # Optional: print(f"Duplicate pruned: {child.value}")
                pass
        
        # Extend with only unique children
        deme.instances.extend(unique_children)
        # deme.instances.extend(children)
    
    return deme
        #--------------------------------------------------------------------------

def run_moses(exemplar: Instance, fitness: FitnessOracle, hyperparams: Hyperparams,
              target: List[bool], csv_path:str, metapop: List[Instance], max_iter: int) -> List[Instance]:
    if max_iter <= 0:
        print("Max iterations limit reached...")
        print(f"\n--- Final Metapopulation ---")
        unique_meta = {inst.value: inst for inst in metapop}.values()
        sorted_meta = sorted(unique_meta, key=lambda x: x.score, reverse=True)
        
        for inst in sorted_meta[:10]:
            # Optional: Add Complexity Penalty here just for display or decision
            print(f"Instance: {inst.value} | Score: {inst.score:.5f}")
        return list(sorted_meta)
    else:
        demes = sample_from_TTable(csv_path, hyperparams, exemplar, exemplar.knobs, target, output_col='O')
        print(f"\n[Iter {max_iter}] Running Variation for {len(demes)} demes centered on: {exemplar.value}...")
        # print(f"Running Variation for {len(demes)} demes...")
        new_demes = []
        for deme in demes:
            deme = run_variation(deme, fitness, hyperparams, target)
            new_demes.append(deme)

        print("\n--- Top Instances from Each Deme ---")
        best_instances = []
        seen_solutions = set()

        for i, deme in enumerate(new_demes):
            if not deme.instances:
                print(f"Deme {i}: Empty")
                continue
            inst = max(deme.instances, key=lambda x: x.score)
            best_instances.append(inst)
            print(f"Deme {i} Best: {inst.value:<30} | Score: {inst.score:.4f}")
        meta_dict = {inst.value: inst for inst in metapop}
        added_count = 0

        for inst in best_instances:
            if inst.value not in meta_dict:
                meta_dict[inst.value] = inst
                added_count += 1
            # If it exists but we found a better score (unlikely purely deterministic, but good practice)
            elif inst.score > meta_dict[inst.value].score:
                meta_dict[inst.value] = inst

        # Convert back to list
        metapop = list(meta_dict.values())
        metapop.sort(key=lambda x: x.score, reverse=True)
        
        print(f"Merged {added_count} new unique instances. Metapop Size: {len(metapop)}")

        new_exemplar = metapop[0]
    
        # Check if we are stuck on the same exemplar as previous round
        if new_exemplar.value == exemplar.value:
            print(f"(!) Stagnation: Best is still {new_exemplar.value}... (Score: {new_exemplar.score:.4f})")
            
            if len(metapop) > 1:
                # STRATEGY: Pick the 2nd best, or a random one from top 5 to force exploration
                # This simulates 'Tabs' logic - don't search the same place twice immediately
                # backup_index = 1 
                # Optional: Randomize slightly to escape local optima
                backup_index = random.randint(1, min(4, len(metapop)-1))
                
                new_exemplar = metapop[backup_index]
                print(f" -> Switching focus to rank {backup_index}: {new_exemplar.value[:20]}... (Score: {new_exemplar.score:.4f})")
                
                # Optional: Temporarily boost mutation to break habits
                # hyperparams.mutation_rate = min(0.8, hyperparams.mutation_rate + 0.1)

        return run_moses(new_exemplar, fitness, hyperparams, target, csv_path, metapop, max_iter - 1)


            # inst = max(deme.instances, key=lambda x: x.score)
            # if inst.value not in seen_solutions:
            #     seen_solutions.add(inst.value)
            #     best_instances.append(inst)
            #     print(f"Deme {i} Best: {inst.value:<40} | Score: {inst.score:.4f}")
            # else:
            #     print(f"Deme {i} Best: [Duplicate] {inst.value} | Score: {inst.score:.4f}")
        
        # metapop.extend(best_instances)
        # new_exemplar = max(best_instances, key=lambda x: x.score)
        # return run_moses(new_exemplar, fitness, hyperparams, target, csv_path, metapop, max_iter - 1)

    



def main(): 
    random.seed(42)
    
    metapop = []
    hyperparams = Hyperparams(mutation_rate=0.3, crossover_rate=0.5, num_generations=20, neighborhood_size=15)
    input, target = load_truth_table("example_data/test_bin.csv", output_col='O') 
    knobs = knobs_from_truth_table(input)
    exemplar = Instance(value=f"(AND A)", id=0, score=0.0, knobs=knobs)
    fitness = FitnessOracle(target)
    exemplar_score = fitness.get_fitness(exemplar)
    exemplar.score = exemplar_score
    print(f"Exemplar: {exemplar.value} | Score: {exemplar.score}")
    metapop.append(exemplar)
    new_metapop = run_moses(exemplar, fitness, hyperparams, target, "example_data/test_bin.csv", metapop, max_iter=15)
    # new_metapop.sort(key=lambda x: x.score, reverse=True)
    # for inst in new_metapop:

    # demes = sample_from_TTable("example_data/test_bin.csv", hyperparams, exemplar, knobs, target, output_col='O')
    # print(f"Running moses for {len(demes)} demes...")
    # new_demes = []
    # for deme in demes:
    #     deme = run_variation(deme, fitness, hyperparams, target)
    #     new_demes.append(deme)

    # print("\n--- Top Instances from Each Deme ---")
    # best_instances = []
    # for i, deme in enumerate(new_demes):
    #     if not deme.instances:
    #         print(f"Deme {i}: Empty")
    #         continue
    #     inst = max(deme.instances, key=lambda x: x.score)
    #     best_instances.append(inst)
    #     print(f"Deme {i} Best: {inst.value:<40} | Score: {inst.score:.4f}")
    


    # k = int(len(demes[0].instances) * 0.1)
    # if k < 1: k = 1

    # k = 7 if len(demes[0].instances) >= 7 else len(demes[0].instances)
    # selected_exemplars = tournament_selection(demes[0], k=k, tournament_size=3)
    # scores = {}
    # for index, deme in enumerate(demes):
    #     deme_score = sum(inst.score for inst in deme.instances)
    #     scores[index] = deme_score

    # best_index = max(scores, key=scores.get)
    # print(f"Best Deme Index: {best_index} with Score: {scores[best_index]:.4f}")

    # selected_exemplars = select_top_k(demes[best_index], k=7)
    # values = [inst.value for inst in selected_exemplars]
    # weights = [inst.score for inst in selected_exemplars]
    # miner = DependencyMiner()
    # print(f"Selected Exemplars for Dependency Mining: {values}")
    # miner.fit(values, weights)
    # print(f"Mining for correlation for {len(values)} exemplars...")
    # correlation = miner.get_meaningful_dependencies()
    # print(f"{'Pair' :>20} | {'Strength' :>6} | {'Confidence' :>8}")
    # print("-" * 50)
    # for row in correlation:
    #     print(f"{row['pair']:>20} | {row['strength']:.4f} | {row['confidence']:.4f}")
    
    # bg = BetaFactorGraph()
    # for row in correlation:
    #     bg.add_dependency_rule(row['pair'], row['strength'], row['confidence'])

    # if len(correlation) > 0:
    #     top_rule = correlation[0]
    #     parts = top_rule['pair'].split(' -- ')
    #     if len(parts) > 0:
    #         root_node = parts[0]
    #         print(f"\nDynamically setting prior for '{root_node}' based on rule: {top_rule['pair']}")
            
    #         bg.set_prior(
    #             root_node, 
    #             stv_strength=top_rule['strength'], 
    #             stv_confidence=top_rule['confidence']
    #         )
    # else:
    #     print("No correlations found. Setting default prior for 'A'.")
    #     bg.set_prior("A", stv_strength=0.5, stv_confidence=0.1)


    # bg.run_evidence_propagation(steps=20)
    # print("\n--- Final STV Results ---")
    # print(f"{'Variable':<10} | {'Strength':<8} | {'Confidence':<10} | {'Counts (a/b)'}")
    # for name, node in bg.nodes.items():
    #     s = node.strength
    #     c = node.confidence
    #     print(f"{name:<10} | {s:.4f}   | {c:.4f}     | {node.alpha:.1f}/{node.beta:.1f}")

    # # bg.visualize()


    # stv_dict = {}
    # for name, node in bg.nodes.items():
    #     # VariationQuantale expects values in the format (Strength, Confidence)
    #     stv_dict[name] = (node.strength, node.confidence)
        
    # print(f"\nExtracted STV Values for Crossover: {stv_dict}")

    # # nodes_for_crossover = [name for name, node in bg.nodes.items()]
    
    # # selected_for_crossover = [inst for inst in selected_exemplars if any(k in inst.value for k in nodes_for_crossover[:1])]
    # # print(f"\nSelected for Crossover (based on top nodes): {[inst.value for inst in selected_for_crossover]}")
    # print(f"\nSelected for Crossover (based on top nodes): {[inst.value for inst in selected_exemplars]}")
    # children = crossTopOne(selected_exemplars, stv_dict, target)
    # for inst in children:
    #     print(f"Child: {inst.value} | Score: {inst.score}")

    # mut_instance_parent = selected_for_crossover[0] if selected_for_crossover[0].score > selected_for_crossover[1].score else selected_for_crossover[1] 
    # mutation = Mutation(mut_instance_parent, stv_dict, hyperparams)
    # mutated_child1 = mutation.execute_additive()
    # mutated_child1.score = fitness.get_fitness(mutated_child1)
    # mutated_child2 = mutation.execute_multiplicative()
    # mutated_child2.score = fitness.get_fitness(mutated_child2)
    # print(f"\nMutation of '{mut_instance_parent.value}' | Score: {mut_instance_parent.score}")
    # print(f"Mutated Instance 1: {mutated_child1.value} | Score: {mutated_child1.score}")
    # print(f"Mutated Instance 2: {mutated_child2.value} | Score: {mutated_child2.score}")
    


    # if len(selected_for_crossover) >= 2:
    #     parent1 = selected_for_crossover[0]
    #     parent2 = selected_for_crossover[1]
        
    #     # Pass the extracted dictionary to the crossover class
    #     crossover = VariationQuantale(parent1, parent2, stv_dict)
        
    #     # Optional: Perform Crossover
    #     child = crossover.execute_crossover()
    #     score = fitness.get_fitness(child)
    #     child.score = score
    #     print(f"\nCrossover between '{parent1.value}' | Score: {parent1.score} and '{parent2.value}' | Score: {parent1.score}")
    #     print(f"Crossover Child: {child.value} | Score: {child.score} | Knobs: {[k.symbol for k in child.knobs]}")
    # else:
    #     print("Not enough instances selected for crossover.")




    # crossover = VariationQuantale()



    


    # program_sketch = "(AND $ $)"
    # ITable = [
    # {"A": True,  "B": True,  "O": True},
    # {"A": True,  "B": False, "O": False},
    # {"A": False, "B": True,  "O": False},
    # {"A": False, "B": False, "O": False},
    # ]

    # deme = initialize_deme(program_sketch, ITable)
    
    # print(deme.to_tree())
    # print("")

    # instances = select_top_k(deme, k=2)
    # print(f"selected instances: {instances}")
    # print("")
    # fg = deme.factor_graph
    # for f in fg.factors:
    #     print(f.name, "->", [v.id for v in f.variables])
    
if __name__ == "__main__":
    main()