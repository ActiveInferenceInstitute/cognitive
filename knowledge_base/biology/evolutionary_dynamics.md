---
type: concept
id: evolutionary_dynamics_001
created: 2024-03-15
modified: 2024-03-15
tags: [evolution, dynamics, mathematics, biology, complexity]
aliases: [evolution-dynamics, evolutionary-processes]
complexity: advanced
processing_priority: 1
semantic_relations:
  - type: foundation
    links:
      - [[population_genetics]]
      - [[natural_selection]]
      - [[evolutionary_game_theory]]
  - type: implements
    links:
      - [[fitness_landscapes]]
      - [[replicator_dynamics]]
      - [[adaptive_dynamics]]
  - type: relates
    links:
      - [[ecological_dynamics]]
      - [[developmental_systems]]
      - [[evolutionary_computation]]
---

# Evolutionary Dynamics

## Overview

Evolutionary dynamics describes the mathematical principles governing evolutionary processes across scales, from molecular to ecological systems. This framework integrates concepts from population genetics, game theory, and complex systems theory.

## Mathematical Framework

### Core Dynamics

The fundamental equations of evolutionary dynamics:

```math
\begin{aligned}
& \text{Replicator Equation:} \\
& \dot{x}_i = x_i(f_i(x) - \bar{f}(x)) \\
& \text{Selection Gradient:} \\
& \nabla_s = \frac{\partial \ln W}{\partial s} \\
& \text{Price Equation:} \\
& \Delta\bar{z} = \text{Cov}(w,z) + \mathbb{E}[w\Delta z]
\end{aligned}
```

### Population Genetics Framework

```math
\begin{aligned}
& \text{Wright-Fisher Process:} \\
& P(i \to j) = \binom{N}{j}\left(\frac{i}{N}\right)^j\left(1-\frac{i}{N}\right)^{N-j} \\
& \text{Moran Process:} \\
& T_{i,i+1} = \frac{i(N-i)f_A}{Nf_T}, T_{i,i-1} = \frac{i(N-i)f_B}{Nf_T} \\
& \text{Fixation Probability:} \\
& \rho = \frac{1}{1 + \sum_{k=1}^{N-1}\prod_{i=1}^k\frac{T_{i,i-1}}{T_{i,i+1}}}
\end{aligned}
```

## Implementation Framework

### 1. Evolutionary Simulator

```python
class EvolutionaryDynamics:
    """Simulates evolutionary dynamics"""
    def __init__(self,
                 population_size: int,
                 fitness_function: Callable,
                 mutation_rate: float):
        self.N = population_size
        self.fitness = fitness_function
        self.mu = mutation_rate
        self.population = self.initialize_population()
        
    def simulate_generation(self) -> np.ndarray:
        """Simulate one generation of evolution"""
        # Selection
        fitness_values = self.compute_fitness()
        selected = self.selection_step(fitness_values)
        
        # Reproduction
        offspring = self.reproduction_step(selected)
        
        # Mutation
        mutated = self.mutation_step(offspring)
        
        self.population = mutated
        return self.population
        
    def compute_fitness(self) -> np.ndarray:
        """Compute fitness of current population"""
        return np.array([
            self.fitness(individual)
            for individual in self.population
        ])
        
    def selection_step(self,
                      fitness_values: np.ndarray) -> np.ndarray:
        """Perform selection based on fitness"""
        probabilities = fitness_values / np.sum(fitness_values)
        selected_indices = np.random.choice(
            len(self.population),
            size=self.N,
            p=probabilities
        )
        return self.population[selected_indices]
        
    def reproduction_step(self,
                         selected: np.ndarray) -> np.ndarray:
        """Perform reproduction with recombination"""
        offspring = []
        for i in range(0, self.N, 2):
            parent1 = selected[i]
            parent2 = selected[i+1]
            child1, child2 = self.recombine(parent1, parent2)
            offspring.extend([child1, child2])
        return np.array(offspring)
        
    def mutation_step(self,
                     offspring: np.ndarray) -> np.ndarray:
        """Apply mutations to offspring"""
        mutation_mask = np.random.random(offspring.shape) < self.mu
        mutations = np.random.normal(
            0, 0.1, size=offspring.shape)
        return offspring + mutation_mask * mutations
```

### 2. Fitness Landscape Navigator

```python
class FitnessLandscape:
    """Navigates fitness landscapes"""
    def __init__(self,
                 dimension: int,
                 ruggedness: float):
        self.dim = dimension
        self.ruggedness = ruggedness
        self.landscape = self.generate_landscape()
        
    def generate_landscape(self) -> Callable:
        """Generate NK fitness landscape"""
        def fitness_function(x: np.ndarray) -> float:
            # Base fitness
            base = np.sum(np.sin(x * self.ruggedness))
            
            # Epistatic interactions
            epistasis = 0
            for i in range(self.dim):
                for j in range(i+1, self.dim):
                    epistasis += np.sin(x[i] * x[j])
                    
            return base + self.ruggedness * epistasis
            
        return fitness_function
        
    def local_gradient(self,
                      position: np.ndarray) -> np.ndarray:
        """Compute local fitness gradient"""
        eps = 1e-6
        gradient = np.zeros_like(position)
        
        for i in range(self.dim):
            pos_eps = position.copy()
            pos_eps[i] += eps
            gradient[i] = (
                self.landscape(pos_eps) -
                self.landscape(position)
            ) / eps
            
        return gradient
        
    def find_local_optimum(self,
                          start_position: np.ndarray,
                          learning_rate: float = 0.01,
                          n_steps: int = 1000) -> np.ndarray:
        """Find local fitness optimum"""
        position = start_position.copy()
        
        for _ in range(n_steps):
            gradient = self.local_gradient(position)
            position += learning_rate * gradient
            
        return position
```

### 3. Evolutionary Game Dynamics

```python
class EvolutionaryGame:
    """Simulates evolutionary game dynamics"""
    def __init__(self,
                 payoff_matrix: np.ndarray,
                 population_size: int):
        self.payoff = payoff_matrix
        self.N = population_size
        self.strategies = np.eye(len(payoff_matrix))
        self.population = self.initialize_population()
        
    def compute_fitness(self,
                       population: np.ndarray) -> np.ndarray:
        """Compute fitness based on game interactions"""
        fitness = np.zeros(len(population))
        
        for i, individual in enumerate(population):
            # Compute average payoff against population
            strategy = self.strategies[individual]
            opponent_dist = np.bincount(
                population, minlength=len(self.payoff)
            ) / self.N
            
            fitness[i] = strategy @ self.payoff @ opponent_dist
            
        return fitness
        
    def update_population(self,
                         steps: int = 1) -> np.ndarray:
        """Update population using replicator dynamics"""
        for _ in range(steps):
            fitness = self.compute_fitness(self.population)
            
            # Replicator equation update
            strategy_counts = np.bincount(
                self.population, minlength=len(self.payoff)
            )
            frequencies = strategy_counts / self.N
            
            avg_fitness = frequencies @ self.payoff @ frequencies
            strategy_fitness = self.payoff @ frequencies
            
            # Update frequencies
            dfreq = frequencies * (
                strategy_fitness - avg_fitness)
            
            frequencies += dfreq
            frequencies /= np.sum(frequencies)
            
            # Sample new population
            self.population = np.random.choice(
                len(self.payoff),
                size=self.N,
                p=frequencies
            )
            
        return self.population
```

## Advanced Concepts

### 1. Multilevel Selection

```math
\begin{aligned}
& \text{Group Selection:} \\
& \Delta\bar{z} = \text{Cov}(W_k,\bar{z}_k) + \mathbb{E}[W_k\Delta\bar{z}_k] \\
& \text{Contextual Analysis:} \\
& w_i = \beta_1z_i + \beta_2\bar{z}_k + \epsilon_i
\end{aligned}
```

### 2. Adaptive Dynamics

```math
\begin{aligned}
& \text{Canonical Equation:} \\
& \frac{d}{dt}x = \frac{1}{2}\mu\sigma^2N(x)\left.\frac{\partial^2W(y,x)}{\partial y^2}\right|_{y=x} \\
& \text{Invasion Fitness:} \\
& S(y,x) = \left.\frac{\partial\ln W(y,x)}{\partial y}\right|_{y=x}
\end{aligned}
```

### 3. Evolutionary Stability

```math
\begin{aligned}
& \text{ESS Condition:} \\
& W(x^*,x^*) > W(x,x^*) \text{ for all } x \neq x^* \\
& \text{Convergence Stability:} \\
& \left.\frac{\partial^2W(y,x)}{\partial y^2}\right|_{y=x=x^*} < 0
\end{aligned}
```

## Applications

### 1. Molecular Evolution
- Sequence evolution models
- Phylogenetic inference
- Molecular clocks

### 2. Ecological Evolution
- Species coevolution
- Host-parasite dynamics
- Resource competition

### 3. Cultural Evolution
- Meme dynamics
- Social learning
- Cultural transmission

## Implementation Considerations

### 1. Numerical Methods
- Stochastic simulation algorithms
- Adaptive timestep integration
- Parallel population updates

### 2. Optimization Techniques
- Gradient-based methods
- Evolutionary algorithms
- Multi-objective optimization

### 3. Visualization Tools
- Phase space plots
- Fitness landscapes
- Phylogenetic trees

## References
- [[nowak_2006]] - "Evolutionary Dynamics: Exploring the Equations of Life"
- [[page_2002]] - "The Structure and Dynamics of Evolutionary Systems"
- [[dieckmann_2000]] - "The Geometry of Ecological Interactions"
- [[rice_2004]] - "Evolutionary Theory: Mathematical and Conceptual Foundations"

## See Also
- [[population_genetics]]
- [[ecological_dynamics]]
- [[developmental_systems]]
- [[evolutionary_game_theory]]
- [[fitness_landscapes]] 