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
      - "[[myrmecology]]"
      - "[[apidology]]"
      - "[[swarm_intelligence_implementation]]"
  - type: implements
    links:
      - "[[code/Things/Ant_Colony/README]]"
      - "[[code/Things/Ant_Colony/AGENTS]]"
  - type: relates
    links:
      - "[[foraging_optimization]]"
      - "[[mathematical_models_social_insects]]"
      - "[[active_inference_social_insects]]"
---

# Implementation Examples Using Social Insect Code

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

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
import numpy as np

def forage(evaporation=0.05, steps=600, n_ants=30, size=21, food_at=lambda t: (15, 15), seed=0):
    """Ants leave the nest at (0, 0), walk with a pheromone bias, and lay trail on the way home."""
    rng = np.random.default_rng(seed)
    pher = np.zeros((size, size))
    pos = np.zeros((n_ants, 2), dtype=int)
    carrying = np.zeros(n_ants, dtype=bool)
    moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    delivered = np.zeros(steps, dtype=int)
    for t in range(steps):
        food = np.array(food_at(t))
        for k in range(n_ants):
            if carrying[k]:                                   # walk home along x then y, laying pheromone
                pher[tuple(pos[k])] += 1.0
                axis = 0 if pos[k][0] > 0 else 1
                pos[k][axis] -= 1
                if not pos[k].any():
                    carrying[k] = False
                    delivered[t] += 1
            else:                                             # random walk weighted by 1 + pheromone
                nxt = np.clip(pos[k] + moves, 0, size - 1)
                w = 1.0 + 5.0 * pher[nxt[:, 0], nxt[:, 1]]
                pos[k] = nxt[rng.choice(4, p=w / w.sum())]
                carrying[k] = bool((pos[k] == food).all())
        pher *= 1.0 - evaporation
    return delivered

delivered = forage()
print("food items delivered:", delivered.sum())
```

### Multi-Colony Competition

See `docs/examples/README.md` for runnable examples.


## Advanced Ant Colony Features

### Dynamic Environment Adaptation

```python
# Dynamic environment: the food source moves at t = 300; compare evaporation rates
move = lambda t: (15, 15) if t < 300 else (3, 17)
for evaporation in (0.0, 0.02, 0.1, 0.3):
    d = np.mean([forage(evaporation=evaporation, steps=600, food_at=move, seed=s) for s in range(5)], axis=0)
    print(f"evaporation {evaporation}: delivered before {d[:300].sum():.1f}, after {d[300:].sum():.1f} (mean of 5 runs)")
```

## Swarm Intelligence Optimization Examples

### Traveling Salesman Problem with Ants

See `docs/examples/README.md` for runnable examples.


### Bee-Inspired Function Optimization

```python
rastrigin = lambda x: 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

def abc_minimise(f, dim=3, n_sources=20, n_iters=300, limit=40, bound=5.12, seed=0):
    """Artificial bee colony on a multimodal test function (employed, onlooker, scout phases)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-bound, bound, (n_sources, dim))
    fx = np.array([f(s) for s in x])
    trials = np.zeros(n_sources, dtype=int)

    def explore(i):
        k, j = rng.choice([m for m in range(n_sources) if m != i]), rng.integers(dim)
        v = x[i].copy()
        v[j] = np.clip(v[j] + rng.uniform(-1, 1) * (x[i, j] - x[k, j]), -bound, bound)
        if f(v) < fx[i]:
            x[i], fx[i], trials[i] = v, f(v), 0
        else:
            trials[i] += 1

    for _ in range(n_iters):
        for i in range(n_sources):
            explore(i)                                         # employed bees
        q = 1.0 / (1.0 + fx)
        for i in rng.choice(n_sources, n_sources, p=q / q.sum()):
            explore(i)                                         # onlookers follow better sources
        for i in np.where(trials > limit)[0]:                  # scouts abandon exhausted sources
            x[i], trials[i] = rng.uniform(-bound, bound, dim), 0
            fx[i] = f(x[i])
    return x[np.argmin(fx)], fx.min()

print(abc_minimise(rastrigin))
```

## Active Inference Integration

### Active Inference Ants

```python
# One-step active inference for a foraging ant: hidden state = food side (L or R).
# Outcomes: food, none, cue_L, cue_R. Actions: go_left, go_right, sniff.
A = {"go_left":  np.array([[0.95, 0.0], [0.05, 1.0], [0, 0], [0, 0]]),   # A[o, s] for each action
     "go_right": np.array([[0.0, 0.95], [1.0, 0.05], [0, 0], [0, 0]]),
     "sniff":    np.array([[0, 0], [0, 0], [0.9, 0.1], [0.1, 0.9]])}
log_pref = np.log(np.array([0.4, 0.1, 0.25, 0.25]))         # prefers finding food

def expected_free_energy(action, q):
    """G = risk + ambiguity = KL[q(o|a) || p(o)] + E_q(s) H[p(o|s, a)]."""
    Aa = A[action]
    qo = Aa @ q
    risk = np.sum(qo[qo > 0] * (np.log(qo[qo > 0]) - log_pref[qo > 0]))
    H = -np.sum(np.where(Aa > 0, Aa * np.log(np.where(Aa > 0, Aa, 1)), 0), axis=0)
    return risk + q @ H

for belief_left in (0.5, 0.6, 0.7, 0.9):
    q = np.array([belief_left, 1 - belief_left])
    G = {a: round(expected_free_energy(a, q), 3) for a in A}
    print(belief_left, G, "->", min(G, key=G.get))
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

## References

- Deneubourg, J.-L., Aron, S., Goss, S., & Pasteels, J. M. (1990). The self-organizing exploratory pattern of the Argentine ant. Journal of Insect Behavior, 3, 159-168.
- Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: Optimization by a colony of cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B, 26, 29-41.
- Dorigo, M., & Stützle, T. (2004). Ant Colony Optimization. MIT Press.
- Karaboga, D., & Basturk, B. (2007). A powerful and efficient algorithm for numerical function optimization: Artificial bee colony (ABC) algorithm. Journal of Global Optimization, 39, 459-471.
- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: A process theory. Neural Computation, 29(1), 1-49.
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.
