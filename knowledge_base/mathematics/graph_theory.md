---
title: Graph Theory
type: mathematical_concept
status: stable
created: 2024-03-15
complexity: intermediate
processing_priority: 1
tags:
- mathematics
- discrete_mathematics
- networks
- combinatorics
semantic_relations:
  - type: foundation_for
    links:
      - "[[probabilistic_graphical_models]]"
      - "[[network_science]]"
      - "[[markov_random_fields]]"
  - type: relates
    links:
      - "[[optimization_theory]]"
      - "[[information_theory]]"
      - "[[complexity_theory]]"
---


# Graph Theory

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Overview

Graph Theory is a branch of mathematics that studies the relationships between pairs of objects. It provides the mathematical foundation for analyzing networks, connections, and structural properties in various domains, from social networks to neural architectures.

## Mathematical Foundation

### Basic Definitions

#### Graph Structure

```math

G = (V,E)

```

where:

- $V$ is vertex set

- $E \subseteq V \times V$ is edge set

#### Graph Properties

```math

\begin{align*}

\text{degree}(v) &= |\{u \in V : (v,u) \in E\}| \\

\text{path}(u,v) &= (v_1,\ldots,v_k), v_1=u, v_k=v \\

\text{distance}(u,v) &= \min\{k : \text{path}(u,v) \text{ has length } k\}

\end{align*}

```

## Implementation

### Graph Data Structure

```python
import numpy as np
from collections import deque

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)]      # illustrative undirected graph
n = 5
A = np.zeros((n, n), dtype=int)
for u, v in edges:
    A[u, v] = A[v, u] = 1
degree = A.sum(axis=1)

def distances(A, source):
    """Breadth-first search: distance(source, v) in edges, -1 if unreachable."""
    dist = np.full(len(A), -1)
    dist[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in np.flatnonzero(A[u]):
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

print(degree, distances(A, 0))
```

### Graph Algorithms

```python
import heapq

def dijkstra(adj, source):
    """adj: {u: [(v, weight), ...]} with non-negative weights."""
    dist = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, []):
            if d + w < dist.get(v, float("inf")):
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    return dist

adj = {0: [(1, 1.0), (2, 4.0)], 1: [(2, 2.0), (3, 6.0)], 2: [(3, 3.0)]}   # illustrative
print(dijkstra(adj, 0))        # {0: 0.0, 1: 1.0, 2: 3.0, 3: 6.0}
```

## Applications

### Network Analysis

#### Centrality Measures

- Degree centrality

- Betweenness centrality

- Eigenvector centrality

- PageRank

#### Community Detection

- Modularity optimization

- Spectral clustering

- Label propagation

### Path Finding

#### Shortest Paths

- Dijkstra's algorithm

- Bellman-Ford algorithm

- Floyd-Warshall algorithm

#### Network Flows

- Maximum flow

- Minimum cut

- Bipartite matching

## Best Practices

### Implementation

1. Choose appropriate representation

1. Optimize for operations

1. Handle special cases

1. Consider scalability

### Algorithm Design

1. Analyze complexity

1. Handle edge cases

1. Optimize memory

1. Consider parallelization

### Validation

1. Test connectivity

1. Verify properties

1. Check invariants

1. Benchmark performance

## Common Issues

### Technical Challenges

1. Large graph processing

1. Dynamic updates

1. Memory constraints

1. Algorithm complexity

### Solutions

1. Sparse representations

1. Incremental updates

1. Distributed processing

1. Approximation algorithms

### Spectral Graph Theory

#### Mathematical Foundations

**Definition** (Graph Laplacian): For a graph $G = (V,E)$ with adjacency matrix $A$, the graph Laplacian is defined as:

$$L = D - A$$

where $D$ is the degree matrix with $D_{ii} = \sum_j A_{ij}$.

**Definition** (Normalized Laplacian): The normalized Laplacian is:

$$\mathcal{L} = D^{-1/2}LD^{-1/2} = I - D^{-1/2}AD^{-1/2}$$

**Theorem** (Spectral Properties): The eigenvalues of the Laplacian satisfy:

1. $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$

1. The multiplicity of eigenvalue 0 equals the number of connected components

1. The second smallest eigenvalue $\lambda_2$ (algebraic connectivity) characterizes connectivity

```python
import numpy as np

# Two triangles joined by one edge (illustrative), then the same graph with the bridge removed
A = np.zeros((6, 6))
for u, v in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
    A[u, v] = A[v, u] = 1

def laplacian(A):
    return np.diag(A.sum(axis=1)) - A

vals, vecs = np.linalg.eigh(laplacian(A))
print("lambda_2 =", vals[1])                       # algebraic connectivity
print("Fiedler partition:", vecs[:, 1] > 0)        # separates the two triangles

A[2, 3] = A[3, 2] = 0
vals = np.linalg.eigvalsh(laplacian(A))
print("components =", np.sum(vals < 1e-10))        # multiplicity of eigenvalue 0
```

### Advanced Graph Algorithms

#### Spectral Partitioning

**Theorem** (Cheeger's Inequality): For any graph partition $(S, \bar{S})$ with conductance $\Phi(S)$, the algebraic connectivity $\lambda_2$ of the normalized Laplacian satisfies:

$$\frac{\lambda_2}{2} \leq \min_S \Phi(S) \leq \sqrt{2\lambda_2}$$

This provides a fundamental connection between spectral properties and graph partitioning quality.

## Related Documentation

- discrete mathematics

- combinatorics

- [[probabilistic_graphical_models]]

- [[network_science]]

- [[complexity_theory]]


## References

- Diestel, R. (2017). Graph Theory (5th ed.). Springer.
- Chung, F. R. K. (1997). Spectral Graph Theory. American Mathematical Society.
- Fiedler, M. (1973). Algebraic connectivity of graphs. Czechoslovak Mathematical Journal, 23(2), 298-305.
- Newman, M. E. J. (2006). Modularity and community structure in networks. Proceedings of the National Academy of Sciences, 103(23), 8577-8582.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
