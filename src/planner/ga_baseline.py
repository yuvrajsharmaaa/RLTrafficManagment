"""
Standard Permutation Genetic Algorithm (GA) Baseline for Route Planning.

Goldberg, D. E. (1989). "Genetic Algorithms in Search, Optimization, and Machine Learning."
Davis, L. (1985). "Applying adaptive planned algorithms to the traveling salesman problem."
Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 973-976.

Native Permutation Representation:
----------------------------------
Operates directly on discrete visit-order permutations of stops:
an array of stop indices `[0, 1, ..., n - 1]` with no continuous random-key mapping needed.

Standard Genetic Operators & Canonical Formulation:
---------------------------------------------------
1. Selection: Tournament Selection (size 3-5, default k=3).
   Picks k candidates uniformly at random; the candidate with the lowest cost (best fitness)
   wins and becomes a parent.
2. Crossover: Order Crossover (OX / OX1, Davis 1985).
   Preserves relative ordering from both parents:
   - Selects two random cut points c1 < c2.
   - Copies slice parent1[c1:c2] into child.
   - Fills remaining positions in circular order starting at c2 using elements from parent2
     that are not already present in the copied slice.
   - Crossover rate: default 0.9.
3. Mutation: Swap Mutation.
   Swaps two randomly chosen stop positions with probability p_m per individual (default: 0.08).
4. Elitism:
   Carries over the top E individuals unchanged each generation (default: E=2).

Documented Hyperparameters:
---------------------------
- Tournament size k: 3 (configurable in [3, 5])
- Crossover probability p_c: 0.9 (Order Crossover OX)
- Mutation probability p_m: 0.08 (Swap mutation, range [0.05, 0.10])
- Elitism count E: 2 (preserves top 2 performers)
- Stagnation patience: 15 generations
- Relative improvement tolerance: 1e-6

Budget & Function Evaluation Parity:
------------------------------------
Uses the exact same `default_budget(dim)` as va_qpso, fixed_beta_qpso, and pso_baseline:
    population_size = max(20, 4 * dim)
    max_generations = max(100, 75 * dim)
    max_restarts = max(5, 5 * dim)
Equal population size and generation budget ensure identical function evaluation capacity
across all algorithms.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .fitness import CongestionLookup, score_route

FitnessFn = Callable[[np.ndarray], float]

# Exact Documented Canonical GA Hyperparameters
DEFAULT_TOURNAMENT_SIZE = 3     # Selection tournament size (3-5)
DEFAULT_CROSSOVER_RATE = 0.90   # Order crossover probability (OX)
DEFAULT_MUTATION_RATE = 0.08    # Swap mutation probability per individual (0.05-0.10)
DEFAULT_ELITISM = 2             # Top individuals carried over unchanged
DEFAULT_PATIENCE = 15           # Stagnation generations before restart
DEFAULT_TOL = 1e-6              # Relative improvement tolerance


def default_budget(dim: int) -> Tuple[int, int, int]:
    """
    Population and generation budget scaled off `dim` (number of stops).
    Matches qpso.py and pso_baseline.py default_budget() exactly.

    Returns:
        (population_size, max_generations, max_restarts)
    """
    return max(20, 4 * dim), max(100, 75 * dim), max(5, 5 * dim)


def _resolve_budget(
    dim: int,
    population_size: Optional[int],
    max_generations: Optional[int],
    max_restarts: Optional[int],
) -> Tuple[int, int, int]:
    """Fill in any budget argument left as None from default_budget(dim)."""
    pop_size, gens, restarts = default_budget(dim)
    return (
        pop_size if population_size is None else population_size,
        gens if max_generations is None else max_generations,
        restarts if max_restarts is None else max_restarts,
    )


def order_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Order Crossover (OX / OX1) for permutation sequencing (Davis, 1985).

    Preserves the relative ordering of cities/stops from both parents without
    creating duplicate stops or missing visits.

    Args:
        parent1: Permutation of shape (n,).
        parent2: Permutation of shape (n,).
        rng: Random number generator.

    Returns:
        (child1, child2): Two valid permutation offspring.
    """
    n = len(parent1)
    if n <= 2:
        return np.copy(parent1), np.copy(parent2)

    cut_indices = rng.choice(n, size=2, replace=False)
    c1, c2 = min(cut_indices), max(cut_indices)

    def _cross(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        child = np.full(n, -1, dtype=int)
        # 1. Copy slice from first parent
        child[c1:c2] = p1[c1:c2]
        in_slice = set(p1[c1:c2])

        # 2. Extract remaining elements from second parent in circular order from c2
        remaining = [p2[(c2 + i) % n] for i in range(n) if p2[(c2 + i) % n] not in in_slice]

        # 3. Fill child in circular order starting at c2
        fill_positions = [(c2 + i) % n for i in range(n - (c2 - c1))]
        for pos, val in zip(fill_positions, remaining):
            child[pos] = val

        return child

    return _cross(parent1, parent2), _cross(parent2, parent1)


def swap_mutation(
    individual: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Swap mutation: randomly selects two distinct stop positions and exchanges them.

    Args:
        individual: Stop permutation of shape (n,).
        rng: Random number generator.

    Returns:
        Mutated permutation copy.
    """
    n = len(individual)
    if n <= 1:
        return np.copy(individual)

    mutated = np.copy(individual)
    i, j = rng.choice(n, size=2, replace=False)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def tournament_selection(
    population: List[np.ndarray],
    scores: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Tournament selection (minimization objective).

    Picks `tournament_size` candidates uniformly at random with replacement;
    the candidate with the minimum cost (lowest route score) wins.

    Args:
        population: List of permutation arrays.
        scores: 1D array of fitness scores corresponding to each individual.
        tournament_size: Number of competitors in the tournament.
        rng: Random number generator.

    Returns:
        Winner permutation array.
    """
    candidates = rng.choice(len(population), size=tournament_size, replace=False)
    winner_idx = candidates[np.argmin(scores[candidates])]
    return population[winner_idx]


def genetic_algorithm(
    dim: int,
    fitness_fn: FitnessFn,
    population_size: Optional[int] = None,
    max_generations: Optional[int] = None,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    elitism: int = DEFAULT_ELITISM,
    seed: Optional[int] = None,
    patience: int = DEFAULT_PATIENCE,
    tol: float = DEFAULT_TOL,
    max_restarts: Optional[int] = None,
    return_history: bool = False,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """
    Run standard permutation Genetic Algorithm to sequence stops.

    Args:
        dim: Number of stops to sequence (permutation length).
        fitness_fn: Function mapping integer permutation array order -> float cost.
        population_size: Swarm/population size (defaults to default_budget(dim)[0]).
        max_generations: Iteration/generation budget (defaults to default_budget(dim)[1]).
        tournament_size: Selection tournament size k in [3, 5] (default: 3).
        crossover_rate: Order crossover probability p_c (default: 0.90).
        mutation_rate: Swap mutation probability p_m per individual (default: 0.08).
        elitism: Top E individuals carried over unchanged (default: 2).
        seed: Random seed for deterministic reproducibility.
        patience: Stagnation generations without improvement before restart.
        tol: Relative tolerance required to count as improvement.
        max_restarts: Maximum consecutive stagnation restarts allowed.

    Returns:
        (best_order, best_score): Optimal stop visitation permutation and fitness score.
    """
    population_size, max_generations, max_restarts = _resolve_budget(
        dim, population_size, max_generations, max_restarts
    )

    rng = np.random.default_rng(seed)

    def fresh_population() -> Tuple[List[np.ndarray], np.ndarray]:
        pop = [rng.permutation(dim) for _ in range(population_size)]
        scs = np.array([fitness_fn(ind) for ind in pop])
        return pop, scs

    population, scores = fresh_population()

    # Elite record tracked across all restart cycles
    best_idx = np.argmin(scores)
    best_order = np.copy(population[best_idx])
    best_score = scores[best_idx]

    iterations_since_improvement = 0
    unproductive_restarts = 0
    history: List[float] = []

    for gen in range(max_generations):
        # 1. Elitism: preserve top E individuals
        sorted_indices = np.argsort(scores)
        new_population: List[np.ndarray] = [
            np.copy(population[sorted_indices[e]]) for e in range(min(elitism, population_size))
        ]

        # 2. Reproduction loop: selection, order crossover, and swap mutation
        while len(new_population) < population_size:
            p1 = tournament_selection(population, scores, tournament_size, rng)
            p2 = tournament_selection(population, scores, tournament_size, rng)

            if rng.uniform(0.0, 1.0) < crossover_rate:
                c1, c2 = order_crossover(p1, p2, rng)
            else:
                c1, c2 = np.copy(p1), np.copy(p2)

            if rng.uniform(0.0, 1.0) < mutation_rate:
                c1 = swap_mutation(c1, rng)
            if rng.uniform(0.0, 1.0) < mutation_rate:
                c2 = swap_mutation(c2, rng)

            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)

        population = new_population[:population_size]
        scores = np.array([fitness_fn(ind) for ind in population])

        current_gen_best_idx = np.argmin(scores)
        current_gen_best_score = scores[current_gen_best_idx]

        if current_gen_best_score < best_score - tol:
            best_score = current_gen_best_score
            best_order = np.copy(population[current_gen_best_idx])
            iterations_since_improvement = 0
            unproductive_restarts = 0
        else:
            iterations_since_improvement += 1

        if return_history:
            history.append(float(best_score))

        # Stagnation restart check (identical budget & parity with QPSO / PSO)
        if iterations_since_improvement >= patience:
            if unproductive_restarts >= max_restarts:
                break
            population, scores = fresh_population()
            iterations_since_improvement = 0
            unproductive_restarts += 1

    if return_history:
        while len(history) < max_generations:
            history.append(float(best_score))
        return best_order, best_score, np.asarray(history, dtype=float)

    return best_order, best_score


def replan(
    stops: List[str],
    distance_matrix: np.ndarray,
    congestion_lookup: CongestionLookup,
    volatility_index: float = 0.5,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    population_size: Optional[int] = None,
    max_generations: Optional[int] = None,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    elitism: int = DEFAULT_ELITISM,
    seed: Optional[int] = None,
    patience: int = DEFAULT_PATIENCE,
    tol: float = DEFAULT_TOL,
    max_restarts: Optional[int] = None,
    return_history: bool = False,
) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, np.ndarray]]:
    """
    Execute standard permutation GA to sequence stops on a frozen state snapshot.

    Provides exact signature parity with `src.planner.qpso.replan()` and
    `pso_baseline.replan()` so that routing experiments, benchmarks, and simulation
    evaluators can compare all four optimizers under identical inputs and fitness.

    Args:
        stops: Stop identifiers mapping to distance_matrix rows/cols.
        distance_matrix: (n, n) matrix of live travel times.
        congestion_lookup: Congestion metrics per edge pair.
        volatility_index: Present for interface compatibility (standard GA is non-adaptive).
        weights: (w1, w2, w3) for travel time, distance, and congestion penalties.
        population_size: Population size (or default_budget(n)[0]).
        max_generations: Generation budget (or default_budget(n)[1]).
        tournament_size: Selection tournament size (default: 3).
        crossover_rate: Order crossover probability (default: 0.90).
        mutation_rate: Swap mutation probability (default: 0.08).
        elitism: Elite individuals carried over (default: 2).
        seed: Random seed.
        patience: Stagnation generations before restart.
        tol: Relative tolerance for improvement.
        max_restarts: Max stagnation restarts.

    Returns:
        (best_order, best_score): Optimal stop visit order permutation and total fitness.
    """
    n = len(stops)
    if distance_matrix.shape != (n, n):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} does not match len(stops)={n}."
        )

    def fitness_fn(order: np.ndarray) -> float:
        return score_route(order, distance_matrix, congestion_lookup, weights)

    return genetic_algorithm(
        dim=n,
        fitness_fn=fitness_fn,
        population_size=population_size,
        max_generations=max_generations,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism=elitism,
        seed=seed,
        patience=patience,
        tol=tol,
        max_restarts=max_restarts,
        return_history=return_history,
    )
