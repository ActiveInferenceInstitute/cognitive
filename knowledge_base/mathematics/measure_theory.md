---
title: Measure Theory
type: concept
status: stable
created: 2024-02-12
tags:
  - mathematics
  - probability
  - foundations
semantic_relations:
  - type: foundation
    links: 
      - [[probability_theory]]
      - [[functional_analysis]]
  - type: relates
    links:
      - [[information_theory]]
      - [[optimization_theory]]
      - [[stochastic_processes]]
---

# Measure Theory

## Overview

Measure theory provides the mathematical foundation for modern probability theory and integration. It enables rigorous treatment of probability spaces, random variables, and stochastic processes in cognitive modeling.

## Core Concepts

### Measure Spaces
```math
(X, Σ, μ)
```
where:
- $X$ is sample space
- $Σ$ is σ-algebra
- $μ$ is measure function

### Lebesgue Measure
```math
λ([a,b]) = b - a
```
where:
- $λ$ is Lebesgue measure
- $[a,b]$ is interval

### Probability Measure
```math
P: Σ → [0,1], P(X) = 1
```
where:
- $P$ is probability measure
- $Σ$ is event space

## Implementation

### Measure Space Implementation

```python
from typing import Set, Callable, Optional, Union
import numpy as np

class MeasureSpace:
    def __init__(self,
                 sample_space: Set,
                 sigma_algebra: Set[Set],
                 measure: Callable[[Set], float]):
        """Initialize measure space.
        
        Args:
            sample_space: Sample space
            sigma_algebra: σ-algebra
            measure: Measure function
        """
        self.space = sample_space
        self.algebra = sigma_algebra
        self.measure = measure
    
    def is_measurable(self,
                     subset: Set) -> bool:
        """Check if subset is measurable.
        
        Args:
            subset: Set to check
            
        Returns:
            is_measurable: Whether subset is in σ-algebra
        """
        return subset in self.algebra
    
    def measure_set(self,
                   subset: Set) -> Optional[float]:
        """Compute measure of subset.
        
        Args:
            subset: Set to measure
            
        Returns:
            measure: Measure value if measurable
        """
        if self.is_measurable(subset):
            return self.measure(subset)
        return None
```

### Probability Space

```python
class ProbabilitySpace(MeasureSpace):
    def __init__(self,
                 sample_space: Set,
                 events: Set[Set],
                 prob_measure: Callable[[Set], float]):
        """Initialize probability space.
        
        Args:
            sample_space: Sample space
            events: Event space (σ-algebra)
            prob_measure: Probability measure
        """
        super().__init__(sample_space, events, prob_measure)
        
        # Verify probability measure properties
        if not np.isclose(self.measure(sample_space), 1.0):
            raise ValueError("Probability measure must sum to 1")
    
    def probability(self,
                   event: Set) -> float:
        """Compute probability of event.
        
        Args:
            event: Event to compute probability for
            
        Returns:
            prob: Probability value
        """
        return self.measure_set(event)
    
    def conditional_probability(self,
                             event_a: Set,
                             event_b: Set) -> float:
        """Compute conditional probability P(A|B).
        
        Args:
            event_a: First event
            event_b: Conditioning event
            
        Returns:
            cond_prob: Conditional probability
        """
        prob_b = self.probability(event_b)
        if prob_b == 0:
            raise ValueError("Conditioning on zero probability event")
        
        # Compute intersection
        intersection = event_a.intersection(event_b)
        return self.probability(intersection) / prob_b
```

### Lebesgue Integration

```python
class LebesgueIntegrator:
    def __init__(self,
                 measure_space: MeasureSpace):
        """Initialize Lebesgue integrator.
        
        Args:
            measure_space: Underlying measure space
        """
        self.space = measure_space
    
    def simple_function_integral(self,
                               function: Callable,
                               partition: List[Set],
                               values: List[float]) -> float:
        """Integrate simple function.
        
        Args:
            function: Simple function
            partition: Partition of space
            values: Function values on partition
            
        Returns:
            integral: Integral value
        """
        integral = 0.0
        
        for set_i, value_i in zip(partition, values):
            if self.space.is_measurable(set_i):
                integral += value_i * self.space.measure_set(set_i)
        
        return integral
    
    def approximate_integral(self,
                           function: Callable,
                           n_points: int = 1000) -> float:
        """Approximate Lebesgue integral.
        
        Args:
            function: Function to integrate
            n_points: Number of partition points
            
        Returns:
            integral: Approximate integral value
        """
        # Create partition
        if isinstance(self.space.space, set):
            points = list(self.space.space)
        else:
            # Assume real interval
            a, b = self.space.space
            points = np.linspace(a, b, n_points)
        
        # Compute integral
        integral = 0.0
        for x in points:
            try:
                value = function(x)
                measure = self.space.measure({x})
                integral += value * measure
            except:
                continue
        
        return integral
```

## Applications

### Random Variables

```python
class RandomVariable:
    def __init__(self,
                 prob_space: ProbabilitySpace,
                 mapping: Callable):
        """Initialize random variable.
        
        Args:
            prob_space: Probability space
            mapping: Measurable function
        """
        self.space = prob_space
        self.mapping = mapping
    
    def expectation(self,
                   n_samples: int = 1000) -> float:
        """Compute expected value.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            expectation: Expected value
        """
        integrator = LebesgueIntegrator(self.space)
        return integrator.approximate_integral(
            self.mapping,
            n_samples
        )
    
    def variance(self,
                n_samples: int = 1000) -> float:
        """Compute variance.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            variance: Variance value
        """
        mean = self.expectation(n_samples)
        
        # Define squared deviation
        def squared_dev(x):
            return (self.mapping(x) - mean)**2
        
        # Compute variance
        integrator = LebesgueIntegrator(self.space)
        return integrator.approximate_integral(
            squared_dev,
            n_samples
        )
```

## Best Practices

### Measure Construction
1. Verify σ-algebra properties
2. Check measure properties
3. Handle edge cases
4. Validate consistency

### Integration
1. Choose appropriate partitions
2. Handle singularities
3. Verify convergence
4. Validate results

### Implementation
1. Use stable numerics
2. Handle infinite values
3. Validate properties
4. Test edge cases

## Common Issues

### Technical Challenges
1. Non-measurable sets
2. Singular measures
3. Infinite measures
4. Convergence issues

### Solutions
1. Careful construction
2. Regular approximation
3. Proper decomposition
4. Numerical stability

## Related Documentation
- [[probability_theory]]
- [[functional_analysis]]
- [[stochastic_processes]]

## References
- [[halmos]] - Measure Theory
- [[rudin]] - Real and Complex Analysis
- [[bogachev]] - Measure Theory
- [[evans_gariepy]] - Measure Theory and Fine Properties of Functions 