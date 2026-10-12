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
      - "[[cognitive/swarm_intelligence]]"
      - "[[biology/evolutionary_dynamics]]"
      - "[[systems/emergence]]"
      - "[[code/Things/Ant_Colony/README|Ant Colony Implementations]]"
---

# Swarm Intelligence Implementation Guide

> **Scope.** The snippets on this page are illustrative sketches of the algorithms; they are not
> part of the `cognitive` package and are not exercised by its tests. The repository's own agents are described in
> [[code/Things/README|Things]], and `Ant_Colony` there is a design note without code.

This document provides practical implementations of swarm intelligence algorithms, focusing on ant colony optimization, particle swarm optimization, and other collective intelligence methods. The implementations demonstrate how simple agent rules can produce complex emergent behavior for optimization and problem-solving.

## 🐜 Ant Colony Optimization (ACO) - Ant-Inspired Algorithms

### Biological Foundations in Ants

Ant colony optimization draws directly from real ant foraging behavior, particularly species like *Lasius niger* (black garden ants) and *Linepithema humile* (Argentine ants). In nature, ants deposit pheromone trails that evaporate over time, creating dynamic pathways that guide colony foraging.

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

def aco_tsp(dist, n_ants=20, n_iters=100, alpha=1.0, beta=2.0, rho=0.1, q=1.0, seed=0):
    """Ant colony optimisation for the TSP: p_ij ~ tau_ij^alpha * eta_ij^beta."""
    rng = np.random.default_rng(seed)
    n = len(dist)
    eta = 1.0 / (dist + np.eye(n))              # heuristic desirability 1/d_ij
    tau = np.ones((n, n))                       # pheromone levels
    best_tour, best_len = None, np.inf
    for _ in range(n_iters):
        tours = []
        for _ in range(n_ants):
            tour = [int(rng.integers(n))]
            while len(tour) < n:
                unvisited = [j for j in range(n) if j not in tour]
                w = tau[tour[-1], unvisited] ** alpha * eta[tour[-1], unvisited] ** beta
                tour.append(unvisited[rng.choice(len(unvisited), p=w / w.sum())])
            tours.append(tour)
        lengths = [sum(dist[t[i], t[(i + 1) % n]] for i in range(n)) for t in tours]
        tau *= 1.0 - rho                        # evaporation
        for t, L in zip(tours, lengths):        # deposit q / L along each tour
            for i in range(n):
                tau[t[i], t[(i + 1) % n]] += q / L
                tau[t[(i + 1) % n], t[i]] += q / L
        k = int(np.argmin(lengths))
        if lengths[k] < best_len:
            best_tour, best_len = tours[k], lengths[k]
    return best_tour, best_len
```

### Traveling Salesman Problem Example
```python
rng = np.random.default_rng(1)
cities = rng.random((12, 2))                    # illustrative random instance
dist = np.linalg.norm(cities[:, None] - cities[None, :], axis=-1)
tour, length = aco_tsp(dist)
print(tour, round(length, 3))
```

## 🐦 Particle Swarm Optimization (PSO)

### PSO Algorithm Implementation
```python
def pso(f, dim, n_particles=30, n_iters=200, w=0.7, c1=1.5, c2=1.5, bounds=(-5.0, 5.0), seed=0):
    """Particle swarm optimisation (minimisation) with inertia and cognitive/social pulls."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(*bounds, size=(n_particles, dim))
    v = np.zeros_like(x)
    pbest, pbest_f = x.copy(), np.array([f(p) for p in x])
    g = pbest[np.argmin(pbest_f)].copy()
    for _ in range(n_iters):
        r1, r2 = rng.random((2, n_particles, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (g - x)
        x = np.clip(x + v, *bounds)
        fx = np.array([f(p) for p in x])
        improved = fx < pbest_f
        pbest[improved], pbest_f[improved] = x[improved], fx[improved]
        g = pbest[np.argmin(pbest_f)].copy()
    return g, pbest_f.min()
```

### Function Optimization Example
```python
sphere = lambda x: float(np.sum(x ** 2))        # f(x) = sum x_i^2, minimum 0 at the origin
best_x, best_f = pso(sphere, dim=5)
print(best_x, best_f)
```

## 🐝 Artificial Bee Colony (ABC) Algorithm - Bee-Inspired Algorithms

### Biological Foundations in Bees

Artificial Bee Colony algorithms are inspired by the foraging behavior of honey bees (*Apis mellifera*), particularly their division of labor between employed bees, onlooker bees, and scout bees. In nature, bees use complex communication systems including the waggle dance to share information about food sources.

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
def abc(f, dim, n_sources=20, n_iters=200, limit=30, bounds=(-5.0, 5.0), seed=0):
    """Artificial bee colony (minimisation): employed, onlooker and scout phases."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(*bounds, size=(n_sources, dim))
    fx = np.array([f(s) for s in x])
    trials = np.zeros(n_sources, dtype=int)

    def try_neighbour(i):
        k = rng.choice([m for m in range(n_sources) if m != i])
        j = rng.integers(dim)
        v = x[i].copy()
        v[j] = np.clip(x[i, j] + rng.uniform(-1, 1) * (x[i, j] - x[k, j]), *bounds)
        fv = f(v)                                # v_ij = x_ij + phi (x_ij - x_kj)
        if fv < fx[i]:
            x[i], fx[i], trials[i] = v, fv, 0
        else:
            trials[i] += 1

    for _ in range(n_iters):
        for i in range(n_sources):               # employed bees
            try_neighbour(i)
        quality = 1.0 / (1.0 + fx)               # fitness of a minimisation objective
        for i in rng.choice(n_sources, size=n_sources, p=quality / quality.sum()):
            try_neighbour(i)                     # onlookers pick source i with p_i ~ fitness_i
        for i in np.where(trials > limit)[0]:    # scouts replace exhausted sources
            x[i] = rng.uniform(*bounds, size=dim)
            fx[i], trials[i] = f(x[i]), 0
    return x[np.argmin(fx)], fx.min()

print(abc(sphere, dim=5))
```

## 📊 Performance Analysis

### Benchmark Functions
See `docs/examples/README.md` for runnable examples.


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

## 📚 References and Extensions

### Core Swarm Intelligence References
- [[cognitive/swarm_intelligence|Swarm Intelligence Theory]]
- [[biology/evolutionary_dynamics|Biological Swarm Systems]]
- [[code/Things/Ant_Colony/README|Ant Colony Implementations]]

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
