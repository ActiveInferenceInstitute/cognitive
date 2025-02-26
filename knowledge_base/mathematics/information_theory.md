---
title: Information Theory
type: concept
status: stable
created: 2024-02-12
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
      - [[communication_theory]]
      - [[complexity_theory]]

---

# Information Theory

## Overview

Information Theory provides a mathematical framework for quantifying, storing, and communicating information. It establishes deep connections between physical entropy, computational complexity, and cognitive processes through concepts like mutual information and free energy.

## Core Concepts

### Entropy
1. **Shannon Entropy**
   ```math
   H(X) = -∑_x p(x)log p(x)
   ```
   where:
   - H is entropy
   - p(x) is probability mass function

2. **Differential Entropy**
   ```math
   h(X) = -∫ f(x)log f(x)dx
   ```
   where:
   - h is differential entropy
   - f(x) is probability density function

3. **Rényi Entropy**
   ```math
   H_α(X) = \frac{1}{1-α}log(∑_x p(x)^α)
   ```
   where:
   - α is order parameter
   - p(x) is probability mass function

### Mutual Information
1. **Discrete Case**
   ```math
   I(X;Y) = ∑_{x,y} p(x,y)log\frac{p(x,y)}{p(x)p(y)}
   ```
   where:
   - I is mutual information
   - p(x,y) is joint distribution
   - p(x), p(y) are marginals

2. **Continuous Case**
   ```math
   I(X;Y) = ∫∫ f(x,y)log\frac{f(x,y)}{f(x)f(y)}dxdy
   ```
   where:
   - f(x,y) is joint density
   - f(x), f(y) are marginal densities

### Divergence Measures
1. **Kullback-Leibler Divergence**
   ```math
   D_KL(P||Q) = ∑_x p(x)log\frac{p(x)}{q(x)}
   ```
   where:
   - P, Q are distributions
   - p(x), q(x) are probabilities

2. **f-Divergence**
   ```math
   D_f(P||Q) = ∑_x q(x)f(\frac{p(x)}{q(x)})
   ```
   where:
   - f is convex function
   - p(x), q(x) are probabilities

## Advanced Concepts

### Information Geometry
1. **Fisher Information Metric**
   ```math
   g_{ij}(θ) = E[-\frac{∂²}{∂θ_i∂θ_j}log p(x|θ)]
   ```
   where:
   - g_{ij} is metric tensor
   - p(x|θ) is probability model

2. **α-Connection**
   ```math
   Γ_{ijk}^{(α)} = E[\frac{∂}{\∂θ_i}log p(x|θ)\frac{∂}{\∂θ_j}log p(x|θ)\frac{∂}{\∂θ_k}log p(x|θ)]
   ```
   where:
   - Γ_{ijk} is connection coefficient
   - α is connection parameter

### Quantum Information
1. **Von Neumann Entropy**
   ```math
   S(ρ) = -Tr(ρlog ρ)
   ```
   where:
   - ρ is density matrix
   - Tr is matrix trace

2. **Quantum Relative Entropy**
   ```math
   S(ρ||σ) = Tr(ρ(log ρ - log σ))
   ```
   where:
   - ρ, σ are density matrices

### Rate Distortion Theory
1. **Rate-Distortion Function**
   ```math
   R(D) = min_{p(y|x): E[d(X,Y)]≤D} I(X;Y)
   ```
   where:
   - R is rate
   - D is distortion
   - d is distortion measure

2. **Distortion-Rate Function**
   ```math
   D(R) = min_{p(y|x): I(X;Y)≤R} E[d(X,Y)]
   ```
   where:
   - D is distortion
   - R is rate
   - d is distortion measure

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

### Coding Theory
1. **Source Coding Theorem**
   ```math
   L* ≥ H(X)
   ```
   where:
   - L* is optimal code length
   - H(X) is source entropy

2. **Channel Coding Theorem**
   ```math
   C = max_{p(x)} I(X;Y)
   ```
   where:
   - C is channel capacity
   - I(X;Y) is mutual information

### Machine Learning
1. **Information Bottleneck**
   ```math
   min_{p(t|x)} I(X;T) - βI(T;Y)
   ```
   where:
   - T is bottleneck variable
   - β is trade-off parameter

2. **Maximum Entropy Principle**
   ```math
   max_p H(X) s.t. E_p[f_i(X)] = c_i
   ```
   where:
   - H is entropy
   - f_i are constraints
   - c_i are constants

### Neural Coding
1. **Neural Information Flow**
   ```math
   I(S;R) = H(R) - H(R|S)
   ```
   where:
   - S is stimulus
   - R is neural response

2. **Population Coding**
   ```math
   I(θ;{r_i}) = H({r_i}) - H({r_i}|θ)
   ```
   where:
   - θ is encoded variable
   - r_i are neural responses

## Advanced Applications

### Network Information Theory
1. **Multiple Access Channel**
   ```math
   R_1 + R_2 ≤ I(X_1,X_2;Y)
   ```
   where:
   - R_i are rates
   - X_i are inputs
   - Y is output

2. **Broadcast Channel**
   ```math
   R_1 ≤ I(X;Y_1|U), R_2 ≤ I(U;Y_2)
   ```
   where:
   - R_i are rates
   - U is auxiliary variable
   - Y_i are outputs

### Quantum Computing
1. **Quantum Channel Capacity**
   ```math
   Q(N) = lim_{n→∞} \frac{1}{n}χ(N^⊗n)
   ```
   where:
   - N is quantum channel
   - χ is Holevo information

2. **Entanglement Measures**
   ```math
   E(ρ) = min_{σ∈SEP} S(ρ||σ)
   ```
   where:
   - ρ is quantum state
   - SEP is separable states

### Biological Systems
1. **Metabolic Networks**
   ```math
   I(M;E) = H(M) - H(M|E)
   ```
   where:
   - M is metabolic state
   - E is environment

2. **Gene Regulatory Networks**
   ```math
   I(G;P) = H(P) - H(P|G)
   ```
   where:
   - G is gene expression
   - P is protein levels

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

## Future Directions

### Emerging Areas
1. **Deep Information Theory**
   - Information bottleneck in deep learning
   - Information flow in neural networks
   - Compression bounds

2. **Quantum Information Processing**
   - Quantum error correction
   - Quantum cryptography
   - Quantum algorithms

### Open Problems
1. **Theoretical Challenges**
   - Non-asymptotic bounds
   - High-dimensional estimation
   - Quantum capacities

2. **Practical Challenges**
   - Efficient estimation
   - Real-time processing
   - Quantum implementation

## Related Topics
1. [[coding_theory|Coding Theory]]
2. [[complexity_theory|Complexity Theory]]
3. [[quantum_computing|Quantum Computing]]
4. [[statistical_inference|Statistical Inference]]

## Related Documentation
- [[thermodynamics]]
- [[statistical_physics]]
- [[free_energy_principle]]
- [[coding_theory]]