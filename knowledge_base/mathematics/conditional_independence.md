---
title: Conditional Independence
type: concept
status: stable
created: 2024-03-15
complexity: intermediate
processing_priority: 1
tags:
  - mathematics
  - probability
  - statistics
  - graphical_models
semantic_relations:
  - type: foundation_for
    links:
      - [[markov_blanket]]
      - [[probabilistic_graphical_models]]
  - type: implements
    links:
      - [[probability_theory]]
      - [[information_theory]]
  - type: relates
    links:
      - [[bayesian_networks]]
      - [[markov_random_fields]]
      - [[causal_inference]]

---

# Conditional Independence

## Overview

Conditional Independence is a fundamental concept in probability theory and statistics that describes when the occurrence of one event provides no information about another event, given knowledge of a third event. This concept is crucial for understanding probabilistic graphical models, Markov blankets, and efficient inference algorithms.

## Mathematical Foundation

### Definition
Two random variables X and Y are conditionally independent given Z if:

```math
P(X,Y|Z) = P(X|Z)P(Y|Z)
```

Equivalently:
```math
P(X|Y,Z) = P(X|Z)
```

### Properties

#### Chain Rule Decomposition
```math
P(X_1,...,X_n) = \prod_{i=1}^n P(X_i|X_{1:i-1})
```

#### D-separation
For nodes X, Y, Z in a Bayesian network:
```math
X \perp\!\!\!\perp Y | Z \iff P(X|Y,Z) = P(X|Z)
```

## Implementation

### Testing Conditional Independence

```python
class ConditionalIndependenceTester:
    def __init__(self,
                 data: np.ndarray,
                 alpha: float = 0.05):
        """Initialize CI tester.
        
        Args:
            data: Data matrix
            alpha: Significance level
        """
        self.data = data
        self.alpha = alpha
    
    def partial_correlation_test(self,
                               x: int,
                               y: int,
                               z: List[int]) -> bool:
        """Test CI using partial correlation.
        
        Args:
            x: First variable index
            y: Second variable index
            z: Conditioning set indices
            
        Returns:
            is_independent: Whether variables are CI
        """
        # Compute partial correlation
        corr = self.compute_partial_correlation(x, y, z)
        
        # Fisher z-transform
        z_score = self.fisher_z_transform(corr, len(self.data))
        
        # Test significance
        return abs(z_score) < stats.norm.ppf(1 - self.alpha/2)
    
    def mutual_information_test(self,
                              x: int,
                              y: int,
                              z: List[int]) -> bool:
        """Test CI using conditional mutual information.
        
        Args:
            x: First variable index
            y: Second variable index
            z: Conditioning set indices
            
        Returns:
            is_independent: Whether variables are CI
        """
        # Estimate conditional mutual information
        cmi = self.estimate_cmi(x, y, z)
        
        # Apply threshold test
        return cmi < self.compute_threshold()
```

### Graphical Model Implementation

```python
class ConditionalIndependenceGraph:
    def __init__(self,
                 n_nodes: int):
        """Initialize CI graph.
        
        Args:
            n_nodes: Number of nodes
        """
        self.n_nodes = n_nodes
        self.adjacency = np.zeros((n_nodes, n_nodes))
        self.separating_sets = {}
    
    def add_edge(self,
                i: int,
                j: int):
        """Add edge between nodes.
        
        Args:
            i: First node
            j: Second node
        """
        self.adjacency[i,j] = 1
        self.adjacency[j,i] = 1
    
    def find_separating_set(self,
                          i: int,
                          j: int) -> Set[int]:
        """Find separating set between nodes.
        
        Args:
            i: First node
            j: Second node
            
        Returns:
            sep_set: Separating set
        """
        # Implement separation set search
        pass
    
    def is_conditionally_independent(self,
                                  i: int,
                                  j: int,
                                  z: Set[int]) -> bool:
        """Check if nodes are conditionally independent.
        
        Args:
            i: First node
            j: Second node
            z: Conditioning set
            
        Returns:
            is_ci: Whether nodes are CI
        """
        return self._check_separation(i, j, z)
```

## Applications

### Structure Learning

#### PC Algorithm
- Start with complete graph
- Remove edges based on CI tests
- Orient remaining edges
- Infer causal structure

#### FCI Algorithm
- Handle latent confounders
- Test ancestral relationships
- Build PAG representation

### Probabilistic Inference

#### Belief Propagation
- Message passing
- Factor graph operations
- Marginal computation

#### Variational Inference
- Mean field approximation
- Factorized distributions
- Evidence lower bound

## Best Practices

### Testing
1. Choose appropriate test
2. Consider sample size
3. Handle multiple testing
4. Validate assumptions

### Implementation
1. Efficient data structures
2. Numerical stability
3. Sparse representations
4. Caching results

### Validation
1. Cross-validation
2. Robustness checks
3. Sensitivity analysis
4. Benchmark comparison

## Common Issues

### Technical Challenges
1. Finite sample effects
2. Curse of dimensionality
3. Computational complexity
4. Numerical precision

### Solutions
1. Regularization
2. Efficient algorithms
3. Approximation methods
4. Robust statistics

## Related Documentation
- [[probability_theory]]
- [[markov_blanket]]
- [[graphical_models]]
- [[causal_inference]] 