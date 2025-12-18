---
title: Swarm Intelligence Implementation Guide
type: implementation
status: stable
created: 2025-01-01
updated: 2025-01-01
tags:
  - swarm_intelligence
  - implementation
  - optimization
  - collective_behavior
semantic_relations:
  - type: implements
    links:
      - [[cognitive/swarm_intelligence]]
      - [[biology/evolutionary_dynamics]]
      - [[systems/emergence]]
      - [[Things/Ant_Colony/|Ant Colony Implementations]]
---

# Swarm Intelligence Implementation Guide

This document provides practical implementations of swarm intelligence algorithms, focusing on ant colony optimization, particle swarm optimization, and other collective intelligence methods. The implementations demonstrate how simple agent rules can produce complex emergent behavior for optimization and problem-solving.

## 🐜 Ant Colony Optimization (ACO) - Ant-Inspired Algorithms

### Biological Foundations in Ants

Ant colony optimization draws directly from real ant foraging behavior, particularly species like *Lasius niger* (black garden ants) and *Linepithema humile* (Argentine ants). In nature, ants deposit pheromone trails that evaporate over time, creating dynamic pathways that guide colony foraging.

#### Ant Foraging Behavior
```python
class RealAntForaging:
    """Model of actual ant foraging behavior"""

    def __init__(self):
        self.pheromone_evaporation_rate = 0.1  # Natural pheromone decay
        self.ant_memory_capacity = 10  # Limited spatial memory
        self.recruitment_threshold = 0.7  # Quality threshold for recruitment

    def ant_decision_process(self, current_location, pheromone_levels, distance_to_nest):
        """Simulate real ant decision-making"""
        # Combine pheromone attraction with distance minimization
        # Include stochastic exploration vs exploitation
        pass
```

#### Mathematical Foundation
The probability that an ant at node $i$ chooses to move to node $j$ is given by:

$$p_{ij}^k = \frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in J_i^k} [\tau_{il}]^\alpha \cdot [\eta_{il}]^\beta}$$

where:
- $\tau_{ij}$ is the pheromone level on edge $(i,j)$
- $\eta_{ij} = 1/d_{ij}$ is the heuristic desirability (inverse distance)
- $\alpha$ controls pheromone importance
- $\beta$ controls heuristic importance
- $J_i^k$ is the set of unvisited nodes for ant $k$

### Core ACO Algorithm
```python
import numpy as np
from typing import List, Tuple, Dict, Callable
import matplotlib.pyplot as plt

class AntColonyOptimization:
    """Ant Colony Optimization for combinatorial optimization problems."""

    def __init__(self,
                 n_ants: int,
                 n_iterations: int,
                 evaporation_rate: float,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 q0: float = 0.9):
        """Initialize ACO algorithm.

        Args:
            n_ants: Number of ants in colony
            n_iterations: Number of algorithm iterations
            evaporation_rate: Pheromone evaporation rate (0 < ρ < 1)
            alpha: Pheromone importance factor
            beta: Heuristic importance factor
            q0: Exploitation vs exploration parameter
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0

        # Algorithm state
        self.pheromone_matrix = None
        self.heuristic_matrix = None
        self.best_solution = None
        self.best_fitness = float('inf')

        # Statistics
        self.fitness_history = []

    def initialize_pheromone(self, n_nodes: int, initial_pheromone: float = 1.0):
        """Initialize pheromone matrix.

        Args:
            n_nodes: Number of nodes in problem
            initial_pheromone: Initial pheromone value
        """
        self.pheromone_matrix = np.full((n_nodes, n_nodes), initial_pheromone)

    def initialize_heuristic(self, distance_matrix: np.ndarray):
        """Initialize heuristic matrix from distance matrix.

        Args:
            distance_matrix: Matrix of distances between nodes
        """
        # Heuristic is inverse of distance (closer nodes more attractive)
        with np.errstate(divide='ignore'):
            self.heuristic_matrix = 1.0 / distance_matrix
        self.heuristic_matrix[distance_matrix == 0] = 0  # Avoid division by zero

    def construct_solution(self, start_node: int = 0) -> List[int]:
        """Construct solution using ant's probabilistic decision rule.

        Args:
            start_node: Starting node for tour

        Returns:
            Solution path as list of node indices
        """
        n_nodes = self.pheromone_matrix.shape[0]
        solution = [start_node]
        visited = set([start_node])

        while len(solution) < n_nodes:
            current_node = solution[-1]
            next_node = self.select_next_node(current_node, visited)
            solution.append(next_node)
            visited.add(next_node)

        return solution

    def select_next_node(self, current_node: int, visited: set) -> int:
        """Select next node using pheromone and heuristic information.

        Args:
            current_node: Current node position
            visited: Set of already visited nodes

        Returns:
            Selected next node
        """
        # Get available nodes (not visited)
        available = [i for i in range(len(self.pheromone_matrix))
                    if i not in visited]

        if not available:
            return current_node  # Should not happen in proper TSP

        # Calculate transition probabilities
        pheromone = self.pheromone_matrix[current_node, available]
        heuristic = self.heuristic_matrix[current_node, available]

        # Combined probability calculation
        combined = (pheromone ** self.alpha) * (heuristic ** self.beta)

        # Handle zero probabilities
        if np.sum(combined) == 0:
            # Uniform random selection
            return np.random.choice(available)

        probabilities = combined / np.sum(combined)

        # Exploitation vs exploration
        if np.random.random() < self.q0:
            # Exploitation: choose best option
            best_idx = np.argmax(probabilities)
            return available[best_idx]
        else:
            # Exploration: probabilistic selection
            return np.random.choice(available, p=probabilities)

    def update_pheromone(self, solutions: List[List[int]], fitnesses: List[float]):
        """Update pheromone matrix based on solution quality.

        Args:
            solutions: List of solutions from all ants
            fitnesses: Corresponding fitness values
        """
        # Evaporation
        self.pheromone_matrix *= (1 - self.evaporation_rate)

        # Deposition
        for solution, fitness in zip(solutions, fitnesses):
            # Calculate pheromone deposit amount
            deposit = 1.0 / fitness if fitness > 0 else 1.0

            # Deposit pheromone on solution edges
            for i in range(len(solution) - 1):
                from_node = solution[i]
                to_node = solution[i + 1]
                self.pheromone_matrix[from_node, to_node] += deposit
                self.pheromone_matrix[to_node, from_node] += deposit  # Symmetric

            # Close the tour (return to start)
            from_node = solution[-1]
            to_node = solution[0]
            self.pheromone_matrix[from_node, to_node] += deposit
            self.pheromone_matrix[to_node, from_node] += deposit

    def optimize(self, distance_matrix: np.ndarray,
                fitness_function: Callable[[List[int]], float]) -> Tuple[List[int], float]:
        """Run ACO optimization.

        Args:
            distance_matrix: Distance matrix for the problem
            fitness_function: Function to evaluate solution fitness

        Returns:
            Best solution and its fitness
        """
        n_nodes = distance_matrix.shape[0]

        # Initialize matrices
        self.initialize_pheromone(n_nodes)
        self.initialize_heuristic(distance_matrix)

        for iteration in range(self.n_iterations):
            # Construct solutions
            solutions = []
            for _ in range(self.n_ants):
                solution = self.construct_solution()
                solutions.append(solution)

            # Evaluate fitness
            fitnesses = [fitness_function(sol) for sol in solutions]

            # Update best solution
            for solution, fitness in zip(solutions, fitnesses):
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = solution.copy()

            # Update pheromone
            self.update_pheromone(solutions, fitnesses)

            # Record statistics
            self.fitness_history.append(self.best_fitness)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Best fitness = {self.best_fitness:.4f}")

        return self.best_solution, self.best_fitness
```

### Traveling Salesman Problem Example
```python
def tsp_fitness(solution: List[int], distance_matrix: np.ndarray) -> float:
    """Calculate TSP tour length (fitness function).

    Args:
        solution: Tour as list of city indices
        distance_matrix: Distance matrix between cities

    Returns:
        Total tour length
    """
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += distance_matrix[solution[i], solution[i + 1]]
    # Close the tour
    total_distance += distance_matrix[solution[-1], solution[0]]
    return total_distance

def create_random_tsp_instance(n_cities: int, max_distance: float = 100) -> np.ndarray:
    """Create random TSP distance matrix.

    Args:
        n_cities: Number of cities
        max_distance: Maximum distance between cities

    Returns:
        Distance matrix
    """
    # Create random coordinates
    coordinates = np.random.rand(n_cities, 2) * max_distance

    # Calculate distance matrix
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])

    return distance_matrix

# Usage example
def solve_tsp_with_aco():
    """Demonstrate ACO on TSP."""
    # Create problem instance
    n_cities = 20
    distance_matrix = create_random_tsp_instance(n_cities)

    # Setup ACO
    aco = AntColonyOptimization(
        n_ants=50,
        n_iterations=100,
        evaporation_rate=0.1,
        alpha=1.0,
        beta=2.0,
        q0=0.9
    )

    # Define fitness function
    fitness_fn = lambda sol: tsp_fitness(sol, distance_matrix)

    # Run optimization
    best_solution, best_fitness = aco.optimize(distance_matrix, fitness_fn)

    print(f"Best TSP tour length: {best_fitness:.2f}")
    print(f"Best tour: {best_solution}")

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(aco.fitness_history)
    plt.xlabel('Iteration')
    plt.ylabel('Best Tour Length')
    plt.title('ACO Convergence on TSP')
    plt.grid(True)
    plt.show()

    return best_solution, best_fitness
```

## 🐦 Particle Swarm Optimization (PSO)

### PSO Algorithm Implementation
```python
class ParticleSwarmOptimization:
    """Particle Swarm Optimization for continuous optimization."""

    def __init__(self,
                 n_particles: int,
                 n_dimensions: int,
                 n_iterations: int,
                 bounds: Tuple[float, float],
                 inertia_weight: float = 0.7,
                 cognitive_coeff: float = 1.4,
                 social_coeff: float = 1.4):
        """Initialize PSO algorithm.

        Args:
            n_particles: Number of particles in swarm
            n_dimensions: Problem dimensionality
            n_iterations: Number of iterations
            bounds: Search space bounds (min, max)
            inertia_weight: Particle inertia (w)
            cognitive_coeff: Cognitive component (c1)
            social_coeff: Social component (c2)
        """
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.w = inertia_weight
        self.c1 = cognitive_coeff
        self.c2 = social_coeff

        # Initialize swarm
        self.positions = self.initialize_positions()
        self.velocities = self.initialize_velocities()
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.full(n_particles, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')

        # Statistics
        self.fitness_history = []

    def initialize_positions(self) -> np.ndarray:
        """Initialize particle positions randomly within bounds."""
        min_bound, max_bound = self.bounds
        return np.random.uniform(min_bound, max_bound,
                               (self.n_particles, self.n_dimensions))

    def initialize_velocities(self) -> np.ndarray:
        """Initialize particle velocities."""
        min_bound, max_bound = self.bounds
        velocity_range = max_bound - min_bound
        return np.random.uniform(-velocity_range, velocity_range,
                               (self.n_particles, self.n_dimensions))

    def update_particle(self, i: int):
        """Update position and velocity of particle i."""
        # Calculate acceleration components
        r1, r2 = np.random.random(2)

        cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
        social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])

        # Update velocity
        self.velocities[i] = (self.w * self.velocities[i] +
                            cognitive_component + social_component)

        # Update position
        self.positions[i] += self.velocities[i]

        # Apply bounds
        self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

    def update_bests(self, fitness_function: Callable[[np.ndarray], float]):
        """Update personal and global bests."""
        for i in range(self.n_particles):
            fitness = fitness_function(self.positions[i])

            # Update personal best
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best_positions[i] = self.positions[i].copy()

            # Update global best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.positions[i].copy()

    def optimize(self, fitness_function: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        """Run PSO optimization.

        Args:
            fitness_function: Function to minimize

        Returns:
            Best position and fitness
        """
        # Initialize global best
        self.update_bests(fitness_function)

        for iteration in range(self.n_iterations):
            # Update all particles
            for i in range(self.n_particles):
                self.update_particle(i)

            # Update bests
            self.update_bests(fitness_function)

            # Record statistics
            self.fitness_history.append(self.global_best_fitness)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Best fitness = {self.global_best_fitness:.6f}")

        return self.global_best_position, self.global_best_fitness
```

### Function Optimization Example
```python
def sphere_function(x: np.ndarray) -> float:
    """Sphere function: f(x) = sum(x_i^2)."""
    return np.sum(x**2)

def rastrigin_function(x: np.ndarray) -> float:
    """Rastrigin function: multimodal optimization test function."""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock function: classic optimization test function."""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def run_pso_optimization():
    """Demonstrate PSO on various test functions."""
    # Test functions and their bounds
    test_functions = [
        (sphere_function, (-5.12, 5.12), "Sphere"),
        (rastrigin_function, (-5.12, 5.12), "Rastrigin"),
        (rosenbrock_function, (-2, 2), "Rosenbrock")
    ]

    # PSO parameters
    n_particles = 50
    n_dimensions = 10
    n_iterations = 100

    results = {}

    for fitness_fn, bounds, name in test_functions:
        print(f"\nOptimizing {name} function...")

        # Initialize PSO
        pso = ParticleSwarmOptimization(
            n_particles=n_particles,
            n_dimensions=n_dimensions,
            n_iterations=n_iterations,
            bounds=bounds
        )

        # Run optimization
        best_position, best_fitness = pso.optimize(fitness_fn)

        print(f"Best fitness: {best_fitness:.6f}")
        print(f"Best position: {best_position}")

        results[name] = {
            'fitness': best_fitness,
            'position': best_position,
            'history': pso.fitness_history
        }

    # Plot convergence comparison
    plt.figure(figsize=(12, 8))
    for name, data in results.items():
        plt.plot(data['history'], label=name)

    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('PSO Convergence on Test Functions')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    return results
```

## 🐝 Artificial Bee Colony (ABC) Algorithm - Bee-Inspired Algorithms

### Biological Foundations in Bees

Artificial Bee Colony algorithms are inspired by the foraging behavior of honey bees (*Apis mellifera*), particularly their division of labor between employed bees, onlooker bees, and scout bees. In nature, bees use complex communication systems including the waggle dance to share information about food sources.

#### Bee Foraging Behavior
```python
class RealBeeForaging:
    """Model of actual bee foraging behavior"""

    def __init__(self):
        self.dance_recruitment = True  # Waggle dance communication
        self.scout_exploration_rate = 0.1  # Proportion of scouts
        self.abandonment_limit = 100  # Failed attempts before abandoning source

    def bee_decision_process(self, bee_type, known_sources, current_source_quality):
        """Simulate real bee decision-making based on type"""
        if bee_type == 'employed':
            # Search around known source
            return self.local_search(current_source_quality)
        elif bee_type == 'onlooker':
            # Choose source based on dance information
            return self.select_source_by_dance(known_sources)
        elif bee_type == 'scout':
            # Random exploration
            return self.random_exploration()
```

#### Mathematical Foundation
The onlooker bee selects a food source probabilistically based on its quality:

$$p_i = \frac{fitness_i}{\sum_{j=1}^{SN} fitness_j}$$

where $fitness_i$ is the quality of food source $i$, and $SN$ is the number of food sources.

New candidate solutions are generated using:

$$v_{ij} = x_{ij} + \phi_{ij} \cdot (x_{kj} - x_{ij})$$

where:
- $v_{ij}$ is the new candidate solution
- $x_{kj}$ is a randomly selected solution ($k \neq j$)
- $\phi_{ij}$ is a random number in [-1, 1]

### ABC Implementation
```python
class ArtificialBeeColony:
    """Artificial Bee Colony optimization algorithm."""

    def __init__(self,
                 n_bees: int,
                 n_dimensions: int,
                 n_iterations: int,
                 bounds: Tuple[float, float],
                 limit: int = 100):
        """Initialize ABC algorithm.

        Args:
            n_bees: Total number of bees (employed + onlooker + scout)
            n_dimensions: Problem dimensionality
            n_iterations: Number of iterations
            bounds: Search space bounds
            limit: Abandonment limit for food sources
        """
        self.n_bees = n_bees
        self.n_employed = n_bees // 2  # Employed bees
        self.n_onlooker = n_bees - self.n_employed  # Onlooker bees
        self.n_dimensions = n_dimensions
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.limit = limit

        # Food sources (solutions)
        self.food_sources = self.initialize_food_sources()
        self.fitness_values = np.full(self.n_employed, float('inf'))
        self.trial_counters = np.zeros(self.n_employed)

        # Best solution tracking
        self.best_solution = None
        self.best_fitness = float('inf')

        # Statistics
        self.fitness_history = []

    def initialize_food_sources(self) -> np.ndarray:
        """Initialize food source positions randomly."""
        min_bound, max_bound = self.bounds
        return np.random.uniform(min_bound, max_bound,
                               (self.n_employed, self.n_dimensions))

    def evaluate_fitness(self, fitness_function: Callable[[np.ndarray], float]):
        """Evaluate fitness of all food sources."""
        for i in range(self.n_employed):
            fitness = fitness_function(self.food_sources[i])
            self.fitness_values[i] = fitness

            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = self.food_sources[i].copy()

    def employed_bee_phase(self):
        """Employed bee phase: search for better food sources."""
        for i in range(self.n_employed):
            # Choose random partner bee (different from i)
            partner = np.random.choice([j for j in range(self.n_employed) if j != i])

            # Generate new candidate solution
            phi = np.random.uniform(-1, 1, self.n_dimensions)
            new_solution = (self.food_sources[i] +
                          phi * (self.food_sources[i] - self.food_sources[partner]))

            # Apply bounds
            new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])

            # Evaluate new solution
            new_fitness = self.fitness_function(new_solution)

            # Greedy selection
            if new_fitness < self.fitness_values[i]:
                self.food_sources[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1

    def onlooker_bee_phase(self):
        """Onlooker bee phase: select food sources probabilistically."""
        # Calculate selection probabilities
        probabilities = self.fitness_values / np.sum(self.fitness_values)

        for _ in range(self.n_onlooker):
            # Select food source probabilistically
            selected = np.random.choice(self.n_employed, p=probabilities)

            # Search around selected food source
            partner = np.random.choice([j for j in range(self.n_employed) if j != selected])

            phi = np.random.uniform(-1, 1, self.n_dimensions)
            new_solution = (self.food_sources[selected] +
                          phi * (self.food_sources[selected] - self.food_sources[partner]))

            # Apply bounds
            new_solution = np.clip(new_solution, self.bounds[0], self.bounds[1])

            # Evaluate new solution
            new_fitness = self.fitness_function(new_solution)

            # Greedy selection
            if new_fitness < self.fitness_values[selected]:
                self.food_sources[selected] = new_solution
                self.fitness_values[selected] = new_fitness
                self.trial_counters[selected] = 0
            else:
                self.trial_counters[selected] += 1

    def scout_bee_phase(self):
        """Scout bee phase: abandon exhausted food sources."""
        for i in range(self.n_employed):
            if self.trial_counters[i] >= self.limit:
                # Abandon food source and become scout
                self.food_sources[i] = self.initialize_food_sources()[0]  # Random new position
                self.trial_counters[i] = 0

    def optimize(self, fitness_function: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        """Run ABC optimization.

        Args:
            fitness_function: Function to minimize

        Returns:
            Best solution and fitness
        """
        self.fitness_function = fitness_function

        # Initial evaluation
        self.evaluate_fitness(fitness_function)

        for iteration in range(self.n_iterations):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()

            # Update best solution
            self.evaluate_fitness(fitness_function)

            # Record statistics
            self.fitness_history.append(self.best_fitness)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Best fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness
```

## 🐾 Swarm Intelligence Applications

### Multi-Objective Optimization
```python
class MultiObjectiveACO:
    """ACO for multi-objective optimization problems."""

    def __init__(self, n_ants: int, n_objectives: int, *args, **kwargs):
        """Initialize multi-objective ACO."""
        super().__init__(n_ants, *args, **kwargs)
        self.n_objectives = n_objectives
        self.pareto_front = []

    def update_pheromone_multiobjective(self, solutions: List[List[int]],
                                      objectives: List[List[float]]):
        """Update pheromone for multiple objectives."""
        # Implementation of multi-objective pheromone update
        # (e.g., using Pareto dominance)
        pass
```

### Dynamic Optimization
```python
class DynamicSwarmOptimization:
    """Swarm algorithm for dynamic optimization problems."""

    def __init__(self, *args, **kwargs):
        """Initialize dynamic swarm algorithm."""
        super().__init__(*args, **kwargs)
        self.change_detector = ChangeDetector()
        self.adaptation_mechanism = AdaptationMechanism()

    def detect_environment_change(self) -> bool:
        """Detect if environment has changed."""
        # Implement change detection logic
        return False

    def adapt_to_change(self):
        """Adapt swarm to environmental changes."""
        if self.detect_environment_change():
            self.reinitialize_population()
            self.adaptation_mechanism.apply_adaptation()
```

## 📊 Performance Analysis

### Benchmark Functions
```python
def ackley_function(x: np.ndarray) -> float:
    """Ackley function: multimodal with many local minima."""
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return term1 + term2 + a + np.exp(1)

def griewank_function(x: np.ndarray) -> float:
    """Griewank function: multimodal optimization problem."""
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_term - prod_term + 1

def run_swarm_comparison():
    """Compare different swarm algorithms on benchmark functions."""
    algorithms = {
        'ACO': lambda: AntColonyOptimization(n_ants=50, n_iterations=100, evaporation_rate=0.1),
        'PSO': lambda: ParticleSwarmOptimization(n_particles=50, n_dimensions=10, n_iterations=100, bounds=(-5, 5)),
        'ABC': lambda: ArtificialBeeColony(n_bees=50, n_dimensions=10, n_iterations=100, bounds=(-5, 5))
    }

    benchmark_functions = [
        (sphere_function, (-5, 5), "Sphere"),
        (ackley_function, (-5, 5), "Ackley"),
        (griewank_function, (-5, 5), "Griewank")
    ]

    results = {}

    for algo_name, algo_factory in algorithms.items():
        print(f"\nRunning {algo_name}...")
        algo_results = {}

        for fitness_fn, bounds, fn_name in benchmark_functions:
            print(f"  Optimizing {fn_name}...")

            if algo_name == 'ACO':
                # ACO needs special setup for continuous optimization
                # (would need continuous ACO variant)
                continue

            algorithm = algo_factory()
            if hasattr(algorithm, 'bounds'):
                algorithm.bounds = bounds

            best_pos, best_fit = algorithm.optimize(fitness_fn)
            algo_results[fn_name] = best_fit

        results[algo_name] = algo_results

    # Print comparison table
    print("\n" + "="*50)
    print("Algorithm Comparison Results")
    print("="*50)
    print("<10"    for fn_name in ["Sphere", "Ackley", "Griewank"]:
        print("<10", end="")
    print()
    print("-"*50)

    for algo_name, algo_results in results.items():
        print("<10", end="")
        for fn_name in ["Sphere", "Ackley", "Griewank"]:
            if fn_name in algo_results:
                print("<10.4f", end="")
            else:
                print("<10", end="")
        print()

    return results
```

## 🔧 Implementation Tools

### Visualization Functions
```python
def plot_swarm_trajectory(swarm_algorithm, title: str = "Swarm Trajectory"):
    """Plot particle trajectories during optimization."""
    # Implementation for visualizing swarm behavior
    pass

def plot_convergence_comparison(algorithms_results: Dict, title: str = "Convergence Comparison"):
    """Compare convergence curves of different algorithms."""
    plt.figure(figsize=(12, 8))

    for algo_name, results in algorithms_results.items():
        if 'history' in results:
            plt.plot(results['history'], label=algo_name)

    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title(title)
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def animate_swarm_evolution(positions_history: List[np.ndarray], bounds: Tuple[float, float]):
    """Create animation of swarm evolution."""
    # Implementation for swarm animation
    pass
```

### Statistical Analysis
```python
def perform_statistical_analysis(results: Dict[str, List[float]]) -> Dict:
    """Perform statistical analysis on optimization results."""
    from scipy import stats

    analysis = {}

    # Basic statistics
    for algo_name, fitness_values in results.items():
        analysis[algo_name] = {
            'mean': np.mean(fitness_values),
            'std': np.std(fitness_values),
            'min': np.min(fitness_values),
            'max': np.max(fitness_values)
        }

    # Statistical tests (if multiple runs available)
    if len(list(results.values())[0]) > 1:
        # ANOVA or Kruskal-Wallis test
        all_values = list(results.values())
        if len(all_values) > 1:
            h_stat, p_value = stats.kruskal(*all_values)
            analysis['statistical_test'] = {
                'test': 'Kruskal-Wallis',
                'statistic': h_stat,
                'p_value': p_value
            }

    return analysis
```

## 🐜🐝 Comparative Analysis: Ant vs Bee Swarm Intelligence

### Algorithmic Differences

| Aspect | Ant Colony Optimization | Artificial Bee Colony | Particle Swarm Optimization |
|--------|-------------------------|----------------------|---------------------------|
| **Communication** | Indirect (pheromones) | Direct (dance language) | Direct (position sharing) |
| **Memory** | Trail-based (persistent) | Individual (ephemeral) | Individual + social |
| **Exploration** | Pheromone-guided | Scout bees | Velocity-based |
| **Exploitation** | Trail reinforcement | Dance recruitment | Local search |
| **Convergence** | Gradual accumulation | Probabilistic selection | Velocity damping |

### Biological Inspirations

#### Ant-Based Systems
- **Trail Formation**: Pheromone deposition creates persistent information gradients
- **Colony Coordination**: Division of labor between scouts and followers
- **Dynamic Adaptation**: Pheromone evaporation enables adaptation to changing environments
- **Parallel Exploration**: Multiple ants explore simultaneously

#### Bee-Based Systems
- **Information Sharing**: Explicit communication through waggle dance
- **Role Specialization**: Distinct roles for employed, onlooker, and scout bees
- **Quality Assessment**: Explicit evaluation of food source profitability
- **Abandonment Strategy**: Systematic abandonment of depleted resources

### Performance Characteristics

```python
class SwarmAlgorithmComparison:
    """Comparative analysis of swarm algorithms"""

    def __init__(self):
        self.algorithms = {
            'ACO': {'strengths': ['Discrete optimization', 'Path finding', 'Robust convergence'],
                   'weaknesses': ['Parameter sensitivity', 'Computationally intensive']},
            'ABC': {'strengths': ['Continuous optimization', 'Simple implementation', 'Few parameters'],
                   'weaknesses': ['Local optima trapping', 'Slow convergence']},
            'PSO': {'strengths': ['Fast convergence', 'Memory efficient', 'Easy parallelization'],
                   'weaknesses': ['Parameter tuning', 'Premature convergence']}
        }

    def recommend_algorithm(self, problem_type: str, constraints: dict) -> str:
        """Recommend appropriate swarm algorithm based on problem characteristics"""
        if problem_type == 'combinatorial':
            return 'ACO'
        elif problem_type == 'continuous':
            return 'PSO' if constraints.get('speed', False) else 'ABC'
        else:
            return 'PSO'  # Default choice
```

### Hybrid Approaches

Combining ant and bee strategies creates more robust optimization algorithms:

```python
class HybridAntBeeOptimization:
    """Hybrid algorithm combining ACO and ABC principles"""

    def __init__(self):
        self.ant_system = AntColonyOptimization(...)
        self.bee_system = ArtificialBeeColony(...)
        self.hybrid_communication = HybridCommunicationProtocol()

    def hybrid_search(self, problem_space):
        """Combined search using both ant trails and bee dances"""
        # Ants lay initial pheromone trails
        ant_solutions = self.ant_system.search(problem_space)

        # Bees use ant trails as initial information
        bee_solutions = self.bee_system.search_with_initial_info(ant_solutions)

        # Hybrid communication improves both systems
        self.hybrid_communication.exchange_information(ant_solutions, bee_solutions)

        return self.select_best_solutions(ant_solutions + bee_solutions)
```

## 🚀 Advanced Applications

### Real-World Optimization Problems
```python
class SwarmOptimizationProblems:
    """Collection of real-world optimization problems."""

    @staticmethod
    def traveling_salesman_problem(distance_matrix: np.ndarray) -> Tuple[Callable, Tuple]:
        """TSP problem formulation."""
        def fitness(tour: List[int]) -> float:
            return sum(distance_matrix[tour[i], tour[(i+1) % len(tour)]] for i in range(len(tour)))

        bounds = (0, len(distance_matrix) - 1)  # Discrete bounds
        return fitness, bounds

    @staticmethod
    def portfolio_optimization(returns: np.ndarray, risks: np.ndarray) -> Tuple[Callable, Tuple]:
        """Portfolio optimization problem."""
        def fitness(weights: np.ndarray) -> float:
            # Minimize risk for given return, or maximize return for given risk
            portfolio_return = np.sum(weights * returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns), weights)))
            # Sharpe ratio maximization
            return -portfolio_return / portfolio_risk if portfolio_risk > 0 else float('inf')

        bounds = (0, 1)  # Weight bounds
        return fitness, bounds

    @staticmethod
    def neural_network_training(data: Tuple[np.ndarray, np.ndarray]) -> Callable:
        """Neural network hyperparameter optimization."""
        X, y = data

        def fitness(params: np.ndarray) -> float:
            # params = [learning_rate, batch_size, hidden_units, ...]
            # Train network with these parameters and return validation loss
            # (Simplified - actual implementation would train a real network)
            return np.random.random()  # Placeholder

        return fitness
```

## 📚 References and Extensions

### Core Swarm Intelligence References
- [[cognitive/swarm_intelligence|Swarm Intelligence Theory]]
- [[biology/evolutionary_dynamics|Biological Swarm Systems]]
- [[Things/Ant_Colony/|Ant Colony Implementations]]

### Algorithm Extensions
- **ACO Variants**: MAX-MIN Ant System, Ant Colony System
- **PSO Variants**: Comprehensive Learning PSO, Quantum PSO
- **ABC Variants**: Modified ABC, Gbest-guided ABC

### Advanced Topics
- **Multi-Swarm Systems**: Multiple interacting swarms
- **Hybrid Algorithms**: Combining different swarm algorithms
- **Dynamic Environments**: Adaptation to changing optimization landscapes
- **Multi-Objective Swarm Optimization**: Pareto-based approaches

---

> **Practical Implementation**: Provides working code for major swarm intelligence algorithms.

---

> **Problem-Solving Power**: Demonstrates how simple rules produce complex optimization behavior.

---

> **Extensible Framework**: Easy to adapt for new optimization problems and domains.

---

> **Research Enablement**: Accelerates development of swarm-based solutions for real problems.
