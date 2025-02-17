---
title: Information Theory
type: concept
status: stable
created: 2024-03-15
complexity: advanced
processing_priority: 1
tags:
  - mathematics
  - information
  - computation
  - probability
semantic_relations:
  - type: foundation_for
    links:
      - [[free_energy_principle]]
      - [[predictive_coding]]
      - [[active_inference]]
  - type: implements
    links:
      - [[probability_theory]]
      - [[statistical_physics]]
      - [[entropy_production]]
  - type: relates
    links:
      - [[thermodynamics]]
      - [[stochastic_processes]]
      - [[coding_theory]]

---

# Information Theory

## Overview

Information Theory provides a mathematical framework for quantifying, storing, and communicating information. It establishes deep connections between physical entropy, computational complexity, and cognitive processes through concepts like mutual information and free energy.

## Mathematical Foundation

### Information Measures

#### Shannon Entropy
```math
H(X) = -\sum_{x \in X} p(x) \log p(x)
```
where:
- $H(X)$ is information entropy
- $p(x)$ is probability distribution

#### Mutual Information
```math
I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
```
where:
- $I(X;Y)$ is mutual information
- $p(x,y)$ is joint distribution

#### Kullback-Leibler Divergence
```math
D_{KL}(P||Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
```

## Implementation

### Information Measures

```python
class InformationMetrics:
    def __init__(self,
                 base: float = 2.0,
                 epsilon: float = 1e-10):
        """Initialize information metrics.
        
        Args:
            base: Logarithm base
            epsilon: Numerical stability constant
        """
        self.base = base
        self.eps = epsilon
    
    def entropy(self,
               probabilities: np.ndarray) -> float:
        """Compute Shannon entropy.
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            H: Entropy value
        """
        # Clean probabilities
        p = np.clip(probabilities, self.eps, 1.0)
        p = p / np.sum(p)
        
        # Compute entropy
        return -np.sum(p * np.log(p) / np.log(self.base))
    
    def mutual_information(self,
                         joint_distribution: np.ndarray) -> float:
        """Compute mutual information.
        
        Args:
            joint_distribution: Joint probability distribution
            
        Returns:
            I: Mutual information value
        """
        # Compute marginals
        p_x = np.sum(joint_distribution, axis=1)
        p_y = np.sum(joint_distribution, axis=0)
        
        # Compute mutual information
        mi = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if joint_distribution[i,j] > self.eps:
                    mi += joint_distribution[i,j] * np.log(
                        joint_distribution[i,j] / (p_x[i] * p_y[j])
                    ) / np.log(self.base)
        
        return mi
    
    def kl_divergence(self,
                     p: np.ndarray,
                     q: np.ndarray) -> float:
        """Compute KL divergence.
        
        Args:
            p: First distribution
            q: Second distribution
            
        Returns:
            kl: KL divergence value
        """
        # Clean distributions
        p = np.clip(p, self.eps, 1.0)
        q = np.clip(q, self.eps, 1.0)
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log(p/q) / np.log(self.base))
```

### Information Dynamics

```python
class InformationDynamics:
    def __init__(self,
                 system_dim: int,
                 noise_strength: float = 0.1):
        """Initialize information dynamics.
        
        Args:
            system_dim: System dimension
            noise_strength: Noise magnitude
        """
        self.dim = system_dim
        self.noise = noise_strength
        self.metrics = InformationMetrics()
        
        # Initialize state
        self.state = np.zeros(system_dim)
        self.history = []
    
    def update_state(self,
                    coupling_matrix: np.ndarray,
                    dt: float = 0.01) -> None:
        """Update system state.
        
        Args:
            coupling_matrix: Interaction matrix
            dt: Time step
        """
        # Deterministic update
        drift = coupling_matrix @ self.state
        
        # Stochastic update
        noise = self.noise * np.random.randn(self.dim)
        
        # Update state
        self.state += dt * drift + np.sqrt(dt) * noise
        self.history.append(self.state.copy())
    
    def compute_transfer_entropy(self,
                               source: int,
                               target: int,
                               delay: int = 1) -> float:
        """Compute transfer entropy.
        
        Args:
            source: Source variable index
            target: Target variable index
            delay: Time delay
            
        Returns:
            te: Transfer entropy value
        """
        # Extract time series
        history = np.array(self.history)
        x = history[:-delay, source]
        y = history[delay:, target]
        y_past = history[:-delay, target]
        
        # Estimate joint distributions
        joint_xy = np.histogram2d(x, y, bins=20)[0]
        joint_xyp = np.histogramdd(
            np.column_stack([x, y, y_past]),
            bins=20
        )[0]
        
        # Compute conditional entropies
        h_y = self.metrics.entropy(
            np.sum(joint_xy, axis=0)
        )
        h_yp = self.metrics.entropy(
            np.sum(joint_xyp, axis=(0,1))
        )
        h_xyp = self.metrics.entropy(
            np.sum(joint_xyp, axis=1)
        )
        h_xyyp = self.metrics.entropy(joint_xyp)
        
        # Compute transfer entropy
        return h_y + h_xyp - h_yp - h_xyyp
```

### Information Processing

```python
class InformationProcessor:
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        """Initialize information processor.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        self.metrics = InformationMetrics()
        
        # Initialize transformations
        self.encoder = np.random.randn(hidden_dim, input_dim)
        self.decoder = np.random.randn(output_dim, hidden_dim)
    
    def process_information(self,
                          input_data: np.ndarray,
                          noise_level: float = 0.1) -> Dict[str, float]:
        """Process information through channel.
        
        Args:
            input_data: Input data
            noise_level: Channel noise level
            
        Returns:
            metrics: Information processing metrics
        """
        # Encode input
        hidden = np.tanh(self.encoder @ input_data)
        
        # Add channel noise
        noisy = hidden + noise_level * np.random.randn(*hidden.shape)
        
        # Decode output
        output = np.tanh(self.decoder @ noisy)
        
        # Compute information metrics
        metrics = {
            'input_entropy': self.metrics.entropy(
                np.histogram(input_data, bins=20)[0]
            ),
            'channel_capacity': self.compute_channel_capacity(
                input_data, output, noise_level
            ),
            'information_loss': self.metrics.kl_divergence(
                np.histogram(input_data, bins=20)[0],
                np.histogram(output, bins=20)[0]
            )
        }
        
        return metrics
    
    def compute_channel_capacity(self,
                               input_data: np.ndarray,
                               output_data: np.ndarray,
                               noise_level: float) -> float:
        """Compute channel capacity.
        
        Args:
            input_data: Input data
            output_data: Output data
            noise_level: Channel noise level
            
        Returns:
            capacity: Channel capacity
        """
        # Estimate mutual information
        joint_dist = np.histogram2d(
            input_data.flatten(),
            output_data.flatten(),
            bins=20
        )[0]
        
        mi = self.metrics.mutual_information(joint_dist)
        
        # Upper bound by noise level
        capacity = min(
            mi,
            0.5 * np.log2(1 + 1/noise_level)
        )
        
        return capacity
```

## Applications

### Physical Systems

#### Thermodynamic Systems
- Entropy production
- Heat dissipation
- Reversible computation
- Maxwell's demon

#### Quantum Systems
- Quantum information
- Entanglement entropy
- Quantum channels
- Decoherence

### Cognitive Systems

#### Neural Information
- Neural coding
- Information integration
- Predictive processing
- Learning dynamics

#### Active Inference
- Free energy principle
- Belief updating
- Action selection
- Model selection

## Best Practices

### Analysis
1. Handle edge cases
2. Validate assumptions
3. Consider noise
4. Track information flow

### Implementation
1. Numerical stability
2. Efficient computation
3. Error handling
4. Proper normalization

### Validation
1. Conservation laws
2. Information bounds
3. Channel capacity
4. Error metrics

## Common Issues

### Technical Challenges
1. Numerical precision
2. Distribution estimation
3. High dimensionality
4. Sampling bias

### Solutions
1. Log-space computation
2. Kernel estimation
3. Dimensionality reduction
4. Bootstrap methods

## Related Documentation
- [[thermodynamics]]
- [[statistical_physics]]
- [[free_energy_principle]]
- [[coding_theory]]