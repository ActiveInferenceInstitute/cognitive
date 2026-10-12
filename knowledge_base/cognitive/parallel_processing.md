---
title: Parallel Processing
type: concept
status: stable
tags:
- parallelism
- computation
- efficiency
aliases:
  - parallel_computation
semantic_relations:
  - type: relates
    links:
      - "[[attention_mechanisms]]"
      - "[[resource_management]]"
      - "[[computational_efficiency]]"
---


# Parallel Processing

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

Parallel processing enables concurrent computation across multiple cognitive modules, sensory modalities, and hierarchical levels. In active inference and predictive processing, parallel mechanisms allow efficient belief updating, attention allocation, and action selection while maintaining coherent global representations.

## Neural Parallelism

### Hierarchical Processing Streams

Parallel processing across cortical hierarchies:

```python
import numpy as np

# Two-level Gaussian hierarchy: level 1 predicts the data y, level 2 predicts level 1.
# Both levels update at the same time from the previous iteration's values.
y, prior_top = 2.0, 0.0                        # illustrative data and top-level prior
mu1, mu2 = 0.0, 0.0
for _ in range(200):
    e1 = y - mu1                               # error at the sensory level
    e2 = mu1 - mu2                             # error between levels
    e3 = mu2 - prior_top                       # error against the top prior
    mu1, mu2 = mu1 + 0.1 * (e1 - e2), mu2 + 0.1 * (e2 - e3)
print(mu1, mu2)
```

## Multisensory Integration

### Parallel Modality Processing

Concurrent processing across sensory modalities:

```python
import numpy as np

# Precision-weighted fusion of Gaussian cues, all modalities combined in one vectorised step
means = np.array([1.0, 1.4, 0.8])             # illustrative visual, auditory, tactile estimates
precisions = np.array([4.0, 1.0, 2.0])
fused_mean = np.sum(precisions * means) / np.sum(precisions)
fused_var = 1.0 / np.sum(precisions)
print(fused_mean, fused_var)
```

## Predictive Processing and Parallelism

### Parallel Prediction and Error Computation

Concurrent prediction and error calculation:

```python
import numpy as np

# Predictions and prediction errors for many modules in one batched operation
rng = np.random.default_rng(0)
n_modules, d_state, d_obs = 5, 3, 4
W = rng.normal(size=(n_modules, d_obs, d_state))      # generative mappings (illustrative)
mu = rng.normal(size=(n_modules, d_state))            # beliefs
obs = rng.normal(size=(n_modules, d_obs))

pred = np.einsum("mos,ms->mo", W, mu)                 # all predictions at once
err = obs - pred                                      # all prediction errors at once
looped = np.stack([obs[m] - W[m] @ mu[m] for m in range(n_modules)])
print(np.allclose(err, looped))
```

## Computational Architectures

### SIMD and MIMD Processing

Different parallel processing architectures:

```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor

x = np.linspace(0, 1, 1000)

# SIMD-style: one operation applied to many data elements
simd = np.exp(-x) * np.sin(10 * x)
loop = np.array([np.exp(-v) * np.sin(10 * v) for v in x])
print(np.allclose(simd, loop))

# MIMD-style: different operations run concurrently on the same data
tasks = [np.mean, np.std, np.max]
with ThreadPoolExecutor(max_workers=3) as pool:
    print([round(float(v), 3) for v in pool.map(lambda f: f(x), tasks)])
```

## Synchronization and Coherence

### Belief Synchronization

Maintaining coherent beliefs across parallel processes:

```python
import numpy as np

# Each parallel process holds a categorical belief; consensus by repeated pooling of neighbours
beliefs = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]])   # illustrative
adjacency = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float)       # who talks to whom
W = adjacency / adjacency.sum(axis=1, keepdims=True)                        # row-stochastic mixing

for _ in range(50):
    beliefs = W @ beliefs                    # linear opinion pool over neighbours
print(beliefs)                               # rows agree: a shared belief
```

## Performance Optimization

### Load Balancing

Distributing computational load across parallel units:

```python
import heapq

def balance(costs, n_workers):
    """Greedy longest-processing-time-first assignment of tasks to workers."""
    heap = [(0.0, w) for w in range(n_workers)]
    heapq.heapify(heap)
    assignment = {w: [] for w in range(n_workers)}
    for task in sorted(range(len(costs)), key=lambda t: -costs[t]):
        load, w = heapq.heappop(heap)
        assignment[w].append(task)
        heapq.heappush(heap, (load + costs[task], w))
    return assignment, max(load for load, _ in heap)

costs = [5, 3, 8, 2, 7, 4]                   # illustrative task costs
print(balance(costs, 3))
```

## Applications and Benefits

### Efficiency Gains

Parallel processing provides multiple advantages:

- **Speed**: Concurrent computation reduces processing time
- **Robustness**: Redundant processing maintains function despite failures
- **Scalability**: Additional processing units improve performance
- **Flexibility**: Different operations can proceed simultaneously

### Biological Relevance

Neural systems extensively use parallel processing:

- **Cortical columns**: Parallel processing units in visual cortex
- **Hemispheric specialization**: Parallel processing in brain hemispheres
- **Distributed memory**: Parallel access to different memory systems
- **Motor coordination**: Parallel control of different muscle groups

### Implementation Challenges

Managing parallel processing complexity:

- **Synchronization**: Coordinating parallel processes
- **Communication overhead**: Cost of information exchange
- **Load balancing**: Optimal resource utilization
- **Fault tolerance**: Handling processing failures

---

## Related Concepts

- [[attention_mechanisms]] - Selective processing mechanisms
- [[resource_management]] - Allocation of computational resources
- [[computational_efficiency]] - Optimization of processing efficiency
- [[hierarchical_processing]] - Multi-level information processing
- [[multisensory_integration]] - Combining information from multiple senses


## References

- Felleman, D. J., & Van Essen, D. C. (1991). Distributed hierarchical processing in the primate cerebral cortex. Cerebral Cortex, 1(1), 1-47.
- Goodale, M. A., & Milner, A. D. (1992). Separate visual pathways for perception and action. Trends in Neurosciences, 15(1), 20-25.
- Stein, B. E., & Stanford, T. R. (2008). Multisensory integration: Current issues from the perspective of the single neuron. Nature Reviews Neuroscience, 9(4), 255-266.
- Flynn, M. J. (1972). Some computer organizations and their effectiveness. IEEE Transactions on Computers, C-21(9), 948-960.
- Friston, K. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society B: Biological Sciences, 360(1456), 815-836.
