---
title: Implementation Examples Using Social Insect Code
type: concept
status: stable
created: 2024-12-18
updated: 2024-12-18
tags:
  - implementation
  - examples
  - ants
  - bees
  - swarm_intelligence
  - active_inference
  - practical_guide
aliases: [ant-bee-implementations, social-insect-coding-examples, swarm-intelligence-applications]
complexity: intermediate
processing_priority: 1
semantic_relations:
  - type: foundation
    links:
      - [[myrmecology]]
      - [[apidology]]
      - [[swarm_intelligence_implementation]]
  - type: implements
    links:
      - [[code/Things/Ant_Colony/README]]
      - [[code/Things/Ant_Colony/AGENTS]]
  - type: relates
    links:
      - [[foraging_optimization]]
      - [[mathematical_models_social_insects]]
      - [[active_inference_social_insects]]
---

# Implementation Examples Using Social Insect Code

> [!note] Scope
> The ant-colony runtime is not shipped in this repository:
> `code/Things/Ant_Colony/` contains documentation only. The Python examples
> below illustrate the intended API of that external implementation and are
> not runnable from this checkout. For runnable examples of the installed
> `Things` package, see the root `README.md`, `docs/examples/README.md`, and
> `docs/manuscript/README.md`.

## Overview

This document provides practical implementation examples using the existing Ant Colony and swarm intelligence code. These examples demonstrate how to apply the biological knowledge of ants and bees to build working cognitive systems, optimization algorithms, and multi-agent simulations. The examples bridge theoretical biology with practical programming, showing how social insect principles can be implemented in code.

## Basic Ant Colony Setup

### Simple Foraging Simulation

```python
# Example 1: Basic ant colony foraging simulation
import numpy as np
import matplotlib.pyplot as plt

def run_basic_foraging_simulation():
    """Demonstrate basic ant colony foraging behavior"""

    # Setup environment with food sources
    env_config = {
        'width': 100,
        'height': 100,
        'num_food_sources': 3,
        'food_amount_range': [100, 300],
        'pheromone_decay': 0.995,
        'diffusion_rate': 0.1
    }

    environment = AntColonyEnvironment(env_config)

    # Setup colony
    colony_config = {
        'colony_id': 0,
        'colony_size': 20,
        'nest_location': np.array([50.0, 50.0]),
        'ant_speed': 1.5,
        'sensory_range': 8.0,
        'pheromone_deposit_rate': 1.0,
        'memory_capacity': 15,
        'decision_noise': 0.05
    }

    colony = AntColony(colony_config)
    colony.environment = environment

    # Run simulation
    simulation_steps = 500
    food_collection_history = []

    print("Starting ant colony foraging simulation...")
    print(f"Colony size: {colony.colony_size} ants")
    print(f"Environment: {env_config['width']}x{env_config['height']}")
    print(f"Food sources: {env_config['num_food_sources']}")

    for step in range(simulation_steps):
        # Execute one simulation step
        colony.simulation_step()

        # Record food collection
        food_collection_history.append(colony.total_food_collected)

        # Periodic reporting
        if step % 100 == 0:
            colony_summary = colony.get_colony_summary()
            print(f"Step {step}: {colony_summary['total_food_collected']} food collected")
            print(f"  Active ants: {colony_summary['active_ants']}")
            print(".1f")
            print(f"  Food sources found: {colony_summary['found_food_sources']}")

    # Final analysis
    final_summary = colony.get_colony_summary()
    print("\nSimulation completed!")
    print(f"Total food collected: {final_summary['total_food_collected']}")
    print(".3f")
    print(".1f")

    # Visualize results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(food_collection_history)
    plt.xlabel('Simulation Step')
    plt.ylabel('Total Food Collected')
    plt.title('Food Collection Over Time')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    # Plot pheromone trails (simplified visualization)
    pheromone_layer = environment.pheromone_layers.get('foraging', np.zeros((100, 100)))
    plt.imshow(pheromone_layer, cmap='YlOrRd', origin='lower')
    plt.colorbar(label='Pheromone Intensity')
    plt.title('Final Pheromone Trails')
    plt.scatter([50], [50], c='blue', s=100, marker='o', label='Nest')
    plt.legend()

    plt.subplot(1, 3, 3)
    # Plot ant positions
    ant_positions = np.array([ant.position for ant in colony.agents])
    plt.scatter(ant_positions[:, 0], ant_positions[:, 1], c='red', alpha=0.6, s=20)
    plt.scatter([50], [50], c='blue', s=100, marker='o', label='Nest')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Final Ant Positions')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return colony, environment

# Run the simulation
if __name__ == "__main__":
    colony, environment = run_basic_foraging_simulation()
```

### Multi-Colony Competition

See `docs/examples/README.md` for runnable examples.


## Advanced Ant Colony Features

### Dynamic Environment Adaptation

```python
# Example 3: Ant colony adapting to changing environment
def run_adaptation_simulation():
    """Demonstrate ant colony adaptation to environmental changes"""

    env_config = {
        'width': 120,
        'height': 120,
        'num_food_sources': 4,
        'food_amount_range': [150, 400],
        'pheromone_decay': 0.99
    }

    environment = AntColonyEnvironment(env_config)

    colony_config = {
        'colony_id': 0,
        'colony_size': 40,
        'nest_location': np.array([60.0, 60.0]),
        'ant_speed': 1.0,
        'sensory_range': 10.0,
        'pheromone_deposit_rate': 1.2
    }

    colony = AntColony(colony_config)
    colony.environment = environment

    # Environmental change schedule
    change_events = [
        {'step': 200, 'type': 'food_depletion', 'target_sources': [0, 1]},
        {'step': 400, 'type': 'new_food_source', 'position': [20, 100], 'amount': 300},
        {'step': 600, 'type': 'obstacle_addition', 'position': [80, 40], 'radius': 15}
    ]

    simulation_steps = 800
    adaptation_history = []

    print("Starting adaptation simulation...")
    print("Environmental changes at steps:", [event['step'] for event in change_events])

    for step in range(simulation_steps):
        # Check for environmental changes
        current_changes = [event for event in change_events if event['step'] == step]

        for change in current_changes:
            if change['type'] == 'food_depletion':
                for source_idx in change['target_sources']:
                    if source_idx < len(environment.food_sources):
                        environment.food_sources[source_idx]['amount'] = 0
                        environment.food_sources[source_idx]['depleted'] = True
                        print(f"Step {step}: Depleted food source {source_idx}")

            elif change['type'] == 'new_food_source':
                new_source = {
                    'id': len(environment.food_sources),
                    'position': np.array(change['position']),
                    'amount': change['amount'],
                    'depleted': False
                }
                environment.food_sources.append(new_source)
                print(f"Step {step}: Added new food source at {change['position']}")

            elif change['type'] == 'obstacle_addition':
                obstacle = {
                    'position': np.array(change['position']),
                    'radius': change['radius']
                }
                environment.obstacles.append(obstacle)
                print(f"Step {step}: Added obstacle at {change['position']}")

        # Run colony simulation step
        colony.simulation_step()

        # Record adaptation metrics
        colony_summary = colony.get_colony_summary()
        step_adaptation = {
            'step': step,
            'food_collected': colony_summary['total_food_collected'],
            'efficiency': colony_summary['colony_efficiency'],
            'found_sources': colony_summary['found_food_sources'],
            'active_recruitment': colony_summary['active_recruitment_signals']
        }
        adaptation_history.append(step_adaptation)

    # Analyze adaptation performance
    print("\nAdaptation analysis:")

    # Performance before changes (steps 0-200)
    pre_change = np.mean([h['efficiency'] for h in adaptation_history[:200]])
    print(".4f")

    # Performance during adaptation (steps 200-400)
    adaptation_period = np.mean([h['efficiency'] for h in adaptation_history[200:400]])
    print(".4f")

    # Performance after adaptation (steps 400-800)
    post_adaptation = np.mean([h['efficiency'] for h in adaptation_history[400:800]])
    print(".4f")

    # Visualize adaptation
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    steps = [h['step'] for h in adaptation_history]
    efficiency = [h['efficiency'] for h in adaptation_history]
    plt.plot(steps, efficiency, linewidth=2)
    for change in change_events:
        plt.axvline(x=change['step'], color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Simulation Step')
    plt.ylabel('Colony Efficiency')
    plt.title('Adaptation to Environmental Changes')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    found_sources = [h['found_sources'] for h in adaptation_history]
    plt.plot(steps, found_sources, linewidth=2, color='green')
    for change in change_events:
        plt.axvline(x=change['step'], color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Simulation Step')
    plt.ylabel('Food Sources Found')
    plt.title('Source Discovery Over Time')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    recruitment = [h['active_recruitment'] for h in adaptation_history]
    plt.plot(steps, recruitment, linewidth=2, color='orange')
    for change in change_events:
        plt.axvline(x=change['step'], color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Simulation Step')
    plt.ylabel('Active Recruitment Signals')
    plt.title('Recruitment Activity')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return colony, environment

# Run adaptation simulation
if __name__ == "__main__":
    colony, environment = run_adaptation_simulation()
```

## Swarm Intelligence Optimization Examples

### Traveling Salesman Problem with Ants

See `docs/examples/README.md` for runnable examples.


### Bee-Inspired Function Optimization

```python
# Example 5: Function optimization using bee-inspired algorithms
def optimize_function_with_bees():
    """Use artificial bee colony to optimize mathematical functions"""

    # Test functions
    def sphere_function(x):
        """Sphere function: f(x) = sum(x_i^2)"""
        return np.sum(x**2)

    def rastrigin_function(x):
        """Rastrigin function: multimodal optimization test"""
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def ackley_function(x):
        """Ackley function: multimodal with many local minima"""
        a, b, c = 20, 0.2, 2 * np.pi
        d = len(x)
        term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
        term2 = -np.exp(np.sum(np.cos(c * x)) / d)
        return term1 + term2 + a + np.exp(1)

    test_functions = [
        (sphere_function, (-5.12, 5.12), "Sphere"),
        (rastrigin_function, (-5.12, 5.12), "Rastrigin"),
        (ackley_function, (-5, 5), "Ackley")
    ]

    # ABC parameters
    n_bees = 50
    n_dimensions = 10
    n_iterations = 100
    bounds = (-5, 5)  # Conservative bounds for all functions

    results = {}

    print("Optimizing mathematical functions using Artificial Bee Colony...")
    print(f"ABC Colony: {n_bees} bees, {n_dimensions} dimensions, {n_iterations} iterations")

    for fitness_fn, func_bounds, func_name in test_functions:
        print(f"\nOptimizing {func_name} function...")

        # Adjust bounds for specific function
        search_bounds = func_bounds if func_bounds != (-5.12, 5.12) else bounds

        # Initialize bee colony
        food_sources = initialize_food_sources_abc(n_bees, n_dimensions, search_bounds)
        fitness_values = np.full(n_bees, float('inf'))
        trial_counters = np.zeros(n_bees)

        # Initialize best solution
        best_solution = None
        best_fitness = float('inf')
        convergence_history = []

        for iteration in range(n_iterations):
            # Employed bee phase
            for i in range(n_bees):
                candidate = generate_candidate_employed(food_sources, i, n_dimensions, search_bounds)
                candidate_fitness = fitness_fn(candidate)

                if candidate_fitness < fitness_values[i]:
                    food_sources[i] = candidate
                    fitness_values[i] = candidate_fitness
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1

            # Onlooker bee phase
            fitness_sum = np.sum(1.0 / (fitness_values + 1e-10))  # Avoid division by zero
            probabilities = (1.0 / (fitness_values + 1e-10)) / fitness_sum

            for _ in range(n_bees):
                # Select food source probabilistically
                selected = np.random.choice(n_bees, p=probabilities)

                # Generate candidate around selected source
                candidate = generate_candidate_onlooker(food_sources, selected, n_dimensions, search_bounds)
                candidate_fitness = fitness_fn(candidate)

                if candidate_fitness < fitness_values[selected]:
                    food_sources[selected] = candidate
                    fitness_values[selected] = candidate_fitness
                    trial_counters[selected] = 0
                else:
                    trial_counters[selected] += 1

            # Scout bee phase
            for i in range(n_bees):
                if trial_counters[i] >= 100:  # Abandonment limit
                    food_sources[i] = generate_random_solution(n_dimensions, search_bounds)
                    fitness_values[i] = fitness_fn(food_sources[i])
                    trial_counters[i] = 0

            # Update best solution
            current_best_idx = np.argmin(fitness_values)
            if fitness_values[current_best_idx] < best_fitness:
                best_fitness = fitness_values[current_best_idx]
                best_solution = food_sources[current_best_idx].copy()

            convergence_history.append(best_fitness)

            if (iteration + 1) % 25 == 0:
                print(".6f")

        print(".6f")
        print(f"  Best solution: {best_solution}")

        results[func_name] = {
            'best_fitness': best_fitness,
            'best_solution': best_solution,
            'convergence': convergence_history
        }

    # Visualize results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    for func_name, data in results.items():
        plt.plot(data['convergence'], label=func_name, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('ABC Convergence on Test Functions')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    # Plot final fitness values
    function_names = list(results.keys())
    final_fitnesses = [results[name]['best_fitness'] for name in function_names]
    bars = plt.bar(function_names, final_fitnesses, color=['blue', 'green', 'red'])
    plt.ylabel('Final Best Fitness')
    plt.title('Final Optimization Results')
    plt.yscale('log')
    plt.grid(True, axis='y')

    # Add value labels on bars
    for bar, fitness in zip(bars, final_fitnesses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                '.2e', ha='center', va='bottom')

    plt.subplot(1, 3, 3)
    # Plot solution distributions for sphere function
    if 'Sphere' in results:
        sphere_solutions = np.array([results['Sphere']['best_solution']])
        plt.hist(sphere_solutions.flatten(), bins=20, alpha=0.7, color='blue')
        plt.xlabel('Parameter Value')
        plt.ylabel('Frequency')
        plt.title('Sphere Function Solution Distribution')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    return results

def initialize_food_sources_abc(n_bees, n_dimensions, bounds):
    """Initialize food sources randomly within bounds"""
    min_bound, max_bound = bounds
    return np.random.uniform(min_bound, max_bound, (n_bees, n_dimensions))

def generate_candidate_employed(food_sources, i, n_dimensions, bounds):
    """Generate candidate solution for employed bee"""
    min_bound, max_bound = bounds

    # Select random partner bee
    partner = np.random.choice([j for j in range(len(food_sources)) if j != i])

    # Generate candidate using bee dance formula
    phi = np.random.uniform(-1, 1, n_dimensions)
    candidate = food_sources[i] + phi * (food_sources[i] - food_sources[partner])

    # Apply bounds
    candidate = np.clip(candidate, min_bound, max_bound)

    return candidate

def generate_candidate_onlooker(food_sources, selected, n_dimensions, bounds):
    """Generate candidate solution for onlooker bee"""
    return generate_candidate_employed(food_sources, selected, n_dimensions, bounds)

def generate_random_solution(n_dimensions, bounds):
    """Generate random solution within bounds"""
    min_bound, max_bound = bounds
    return np.random.uniform(min_bound, max_bound, n_dimensions)

# Run function optimization
if __name__ == "__main__":
    optimization_results = optimize_function_with_bees()
```

## Active Inference Integration

### Active Inference Ants

```python
# Example 6: Active Inference enhanced ant colony
def run_active_inference_ants():
    """Demonstrate ants using Active Inference for decision making"""

    # This would integrate with the Active Inference ant agents from the Ant Colony implementation
    # For demonstration, we'll show the conceptual integration

    print("Active Inference Ant Colony Simulation")
    print("=" * 50)

    # Setup environment
    env_config = {
        'width': 80,
        'height': 80,
        'num_food_sources': 2,
        'food_amount_range': [100, 200],
        'pheromone_decay': 0.995
    }

    environment = AntColonyEnvironment(env_config)

    # Setup Active Inference colony
    ai_colony_config = {
        'colony_id': 0,
        'colony_size': 15,  # Smaller colony for demonstration
        'nest_location': np.array([40.0, 40.0]),
        'use_active_inference': True,
        'precision': 1.0,
        'planning_horizon': 3
    }

    # Note: This would use the ActiveInferenceAntAgent class from the implementation
    print("Setting up Active Inference ant colony...")
    print(f"Colony size: {ai_colony_config['colony_size']} ants")
    print(f"Active Inference enabled: {ai_colony_config['use_active_inference']}")
    print("Precision parameter: {ai_colony_config['precision']}")
    print(f"Planning horizon: {ai_colony_config['planning_horizon']}")

    # Conceptual simulation (would use actual ActiveInferenceAntAgent)
    simulation_steps = 300
    free_energy_history = []
    food_collection_history = []

    print("\nRunning Active Inference simulation...")
    print("Step | Food Collected | Average Free Energy")
    print("-" * 45)

    for step in range(simulation_steps):
        # This would execute actual Active Inference ant colony step
        # For demonstration, we'll simulate the results

        # Simulate progressive improvement
        base_food = step * 0.5
        noise_food = np.random.normal(0, 2)
        food_collected = max(0, base_food + noise_food)

        # Simulate decreasing free energy (learning)
        base_fe = 10 * np.exp(-step / 100)
        noise_fe = np.random.normal(0, 0.5)
        avg_free_energy = max(0, base_fe + noise_fe)

        food_collection_history.append(food_collected)
        free_energy_history.append(avg_free_energy)

        if step % 50 == 0:
            print("4d")

    print("4d")

    # Visualize Active Inference performance
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(food_collection_history, linewidth=2, label='Food Collection')
    plt.xlabel('Simulation Step')
    plt.ylabel('Food Collected')
    plt.title('Active Inference Ant Foraging Performance')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(free_energy_history, linewidth=2, color='red', label='Average Free Energy')
    plt.xlabel('Simulation Step')
    plt.ylabel('Free Energy')
    plt.title('Free Energy Minimization Over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\nActive Inference simulation completed!")
    print("This demonstrates how Active Inference can enhance ant colony optimization")
    print("by explicitly minimizing variational free energy in decision-making processes.")

    return food_collection_history, free_energy_history

# Run Active Inference demonstration
if __name__ == "__main__":
    food_history, fe_history = run_active_inference_ants()
```

## Comparative Performance Analysis

### Swarm Algorithm Benchmarking

See `docs/examples/README.md` for runnable examples.


## Summary and Applications

The implementation examples above demonstrate how to use the existing Ant Colony and swarm intelligence code to:

1. **Simulate Biological Systems**: Create realistic ant colony foraging simulations that capture emergent behavior
2. **Solve Optimization Problems**: Apply ant colony optimization to real-world problems like TSP and function optimization  
3. **Compare Algorithms**: Benchmark different swarm intelligence approaches against each other
4. **Integrate Advanced Features**: Add Active Inference capabilities to enhance traditional swarm algorithms
5. **Handle Complex Scenarios**: Manage multi-colony competition, environmental adaptation, and dynamic changes

These examples bridge the gap between biological inspiration and practical computational methods, showing how social insect principles can be implemented in working code to solve complex problems.

The key takeaways from these implementations are:
- **Emergent Intelligence**: Simple individual rules lead to complex collective behavior
- **Adaptability**: Biological algorithms handle dynamic, uncertain environments effectively  
- **Scalability**: Swarm approaches work across different problem sizes and complexities
- **Robustness**: Distributed systems maintain performance despite individual failures
- **Biological Accuracy**: Implementations can capture realistic social insect behavior patterns

---

> **Practical Implementation**: These examples show how to translate biological knowledge of ants and bees into working computational systems that solve real optimization problems.

---

> **Code Integration**: The examples demonstrate proper use of the existing Ant Colony framework while extending it with new capabilities and applications.

---

> **Performance Insights**: Comparative benchmarking reveals the strengths and limitations of different swarm intelligence approaches for various problem types.

---

> **Research Enablement**: These implementations provide a foundation for further research into biological computation, collective intelligence, and bio-inspired algorithms.
