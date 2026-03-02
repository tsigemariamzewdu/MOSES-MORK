import random
from .representation import Deme, Instance
from typing import List

def select_top_k(deme: Deme, k: int) -> List[Instance]:
    """
    select_top_k: Selects the top k instances from the deme based on their scores.
    Args:
        deme (Deme): The deme containing instances.
        k (int): The number of top instances to select.
    Returns: A list of the top k instances.
    """
    sorted_instances = sorted(deme.instances, key=lambda inst: inst.score, reverse=True)
    return sorted_instances[:k]

def tournament_selection(deme: Deme, k:int, tournament_size: int) -> List[Instance]:
    """
    tournament_selection: Selects instances from the deme using tournament selection.
    Args:
        deme (Deme): The deme containing instances.
        k (int): The number of instances to select.
        tournament_size (int): The size of each tournament.
    Returns: A list of selected instances.
    """
    selected_instances = []
    num_instances = len(deme.instances)

    K = min(k, num_instances)
    for _ in range(K):
        tournament = random.sample(deme.instances, min(tournament_size, num_instances))
        winner = max(tournament, key=lambda inst: inst.score)
        selected_instances.append(winner)

    return selected_instances


def boltzmann_roulette_selection(
    deme: Deme,
    k: int,
    temperature: float,
    elite_count: int = 1,
) -> List[Instance]:
    """
    Boltzmann roulette-wheel selection without replacement.

    - Higher score means higher selection probability.
    - Higher temperature increases exploration (flatter distribution).
    - Lower temperature increases exploitation (peaked on top scores).
    - Optional elites are copied first to guarantee best candidates survive.
    """
    if not deme.instances or k <= 0:
        return []

    sorted_instances = sorted(deme.instances, key=lambda inst: inst.score, reverse=True)
    n = len(sorted_instances)
    K = min(k, n)

    elite_count = max(0, min(elite_count, K))
    selected = sorted_instances[:elite_count]

    if len(selected) >= K:
        return selected

    remaining = sorted_instances[elite_count:]

    if temperature <= 1e-12:
        selected.extend(remaining[: (K - len(selected))])
        return selected

    while remaining and len(selected) < K:
        max_score = max(inst.score for inst in remaining)
        weights = [pow(2.718281828459045, (inst.score - max_score) / temperature) for inst in remaining]
        total = sum(weights)

        if total <= 0:
            selected.extend(remaining[: (K - len(selected))])
            break

        pick = random.random() * total
        cumulative = 0.0
        chosen_index = 0
        for i, w in enumerate(weights):
            cumulative += w
            if pick <= cumulative:
                chosen_index = i
                break

        selected.append(remaining.pop(chosen_index))

    return selected