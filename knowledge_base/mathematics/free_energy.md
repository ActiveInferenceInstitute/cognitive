---
type: mathematical_concept
id: free_energy_001
created: 2024-02-05
modified: 2024-03-15
tags: [mathematics, active-inference, free-energy, variational-inference, optimization]
aliases: [variational-free-energy, VFE, evidence-lower-bound, ELBO]
complexity: advanced
processing_priority: 1
semantic_relations:
  - type: implements
    links: 
      - [[active_inference]]
      - [[variational_inference]]
  - type: mathematical_basis
    links:
      - [[information_theory]]
      - [[probability_theory]]
      - [[optimization_theory]]
  - type: relates
    links:
      - [[expected_free_energy]]
      - [[path_integral_free_energy]]
      - [[predictive_coding]]
---

# Free Energy Computation

## What Makes Something a Free Energy?

At its core, a free energy is a functional (a function of functions) that measures the "energetic cost" of the mismatch between two probability distributions - typically between an approximate posterior distribution and the true distribution we're trying to model. The term "free energy" draws inspiration from statistical physics, where it represents the energy available to do useful work in a system.

Key characteristics that define a free energy functional:

1. Variational Form
   - Always involves an expectation over a variational distribution
   - Contains terms measuring both accuracy and complexity
   - Provides a tractable bound on an intractable quantity

2. Information-Theoretic Properties
   - Related to KL divergences between distributions
   - Measures information content and uncertainty
   - Balances model fit against model complexity

3. Optimization Characteristics
   - Serves as an objective function for inference
   - Has well-defined gradients
   - Minimization improves model fit

4. Thermodynamic Analogies
   - Similar structure to physical free energies
   - Trade-off between energy and entropy
   - Equilibrium at minimum free energy

## Mathematical Framework

### Core Definition
The variational free energy $F$ is defined as:

$F = \mathbb{E}_{Q(s)}[\ln Q(s) - \ln P(o,s)]$

where:
- $Q(s)$ is the variational distribution over hidden states
- $P(o,s)$ is the generative model
- $\mathbb{E}_{Q(s)}$ denotes expectation under $Q$

### Alternative Formulations

#### Evidence Lower Bound (ELBO)
$F = -\text{ELBO} = -\mathbb{E}_{Q(s)}[\ln P(o|s)] + \text{KL}[Q(s)||P(s)]$

#### Prediction Error Form
$F = \frac{1}{2}\epsilon^T\Pi\epsilon + \frac{1}{2}\ln|\Sigma| + \text{const}$

where:
- $\epsilon$ is the prediction error
- $\Pi$ is the precision matrix
- $\Sigma$ is the covariance matrix

### Hierarchical Extension
For L-level hierarchical models:

$F = \sum_{l=1}^L \mathbb{E}_{Q(s^{(l)})}[\ln Q(s^{(l)}) - \ln P(s^{(l-1)}|s^{(l)}) - \ln P(s^{(l)}|s^{(l+1)})]$

## Components

### 1. Accuracy Term
- Measures model fit
- [[prediction_error]] minimization
- Likelihood maximization
- Precision weighting

### 2. Complexity Term
- Prior divergence
- [[kl_divergence]] penalty
- Model regularization
- Complexity control

### 3. Entropy Term
- Uncertainty quantification
- Information gain
- Exploration drive
- Posterior sharpness

## Advanced Implementation

### 1. Precision-Weighted Computation
```python
class PrecisionWeightedFreeEnergy:
    def __init__(self):
        self.components = {
            'precision': PrecisionEstimator(
                method='empirical',
                adaptation='online'
            ),
            'error': ErrorComputer(
                type='hierarchical',
                weighting='precision'
            ),
            'complexity': ComplexityComputer(
                method='kl',
                approximation='gaussian'
            )
        }
    
    def compute(
        self,
        beliefs: np.ndarray,
        observations: np.ndarray,
        model: dict,
        precision: np.ndarray
    ) -> Tuple[float, dict]:
        """Compute precision-weighted free energy"""
        # Estimate precision
        pi = self.components['precision'].estimate(
            observations, beliefs)
            
        # Compute prediction errors
        errors = self.components['error'].compute(
            observations, beliefs, model, pi)
            
        # Compute complexity
        complexity = self.components['complexity'].compute(
            beliefs, model['prior'])
            
        # Combine terms
        free_energy = 0.5 * np.sum(errors * pi * errors) + complexity
        
        metrics = {
            'error_term': errors,
            'complexity_term': complexity,
            'precision': pi
        }
        
        return free_energy, metrics
```

### 2. Hierarchical Computation
```python
class HierarchicalFreeEnergy:
    def __init__(self, levels: int):
        self.levels = levels
        self.components = {
            'level_energy': LevelEnergyComputer(
                method='variational',
                coupling='full'
            ),
            'level_coupling': LevelCoupling(
                type='bidirectional',
                strength='adaptive'
            ),
            'total_energy': TotalEnergyComputer(
                method='sum',
                weights='precision'
            )
        }
    
    def compute_hierarchy(
        self,
        beliefs: List[np.ndarray],
        observations: List[np.ndarray],
        models: List[dict]
    ) -> Tuple[float, dict]:
        """Compute hierarchical free energy"""
        # Compute level-wise energies
        level_energies = [
            self.components['level_energy'].compute(
                beliefs[l], observations[l], models[l]
            )
            for l in range(self.levels)
        ]
        
        # Compute level couplings
        couplings = self.components['level_coupling'].compute(
            beliefs, models)
            
        # Compute total energy
        total_energy = self.components['total_energy'].compute(
            level_energies, couplings)
            
        return total_energy
```

### 3. Gradient Computation
```python
class FreeEnergyGradients:
    def __init__(self):
        self.components = {
            'natural': NaturalGradient(
                metric='fisher',
                regularization=True
            ),
            'euclidean': EuclideanGradient(
                method='automatic',
                clipping=True
            ),
            'optimization': GradientOptimizer(
                method='adam',
                learning_rate='adaptive'
            )
        }
    
    def compute_gradients(
        self,
        beliefs: np.ndarray,
        free_energy: float,
        model: dict
    ) -> Tuple[np.ndarray, dict]:
        """Compute free energy gradients"""
        # Natural gradients
        natural_grads = self.components['natural'].compute(
            beliefs, free_energy, model)
            
        # Euclidean gradients
        euclidean_grads = self.components['euclidean'].compute(
            beliefs, free_energy, model)
            
        # Optimize gradients
        final_grads = self.components['optimization'].process(
            natural_grads, euclidean_grads)
            
        return final_grads
```

## Advanced Concepts

### 1. Geometric Properties
- [[information_geometry]]
  - Fisher metrics
  - Natural gradients
- [[wasserstein_geometry]]
  - Optimal transport
  - Geodesic flows

### 2. Variational Methods
- [[mean_field_theory]]
  - Factorized approximations
  - Coordinate descent
- [[bethe_approximation]]
  - Cluster expansions
  - Message passing

### 3. Stochastic Methods
- [[monte_carlo_free_energy]]
  - Importance sampling
  - MCMC methods
- [[path_integral_methods]]
  - Trajectory sampling
  - Action minimization

## Applications

### 1. Inference
- [[state_estimation]]
  - Filtering
  - Smoothing
- [[parameter_estimation]]
  - System identification
  - Model learning

### 2. Learning
- [[model_selection]]
  - Structure learning
  - Complexity control
- [[representation_learning]]
  - Feature extraction
  - Dimensionality reduction

### 3. Control
- [[optimal_control]]
  - Policy optimization
  - Trajectory planning
- [[adaptive_control]]
  - Online adaptation
  - Robust control

## Research Directions

### 1. Theoretical Extensions
- [[quantum_free_energy]]
  - Quantum fluctuations
  - Entanglement effects
- [[relativistic_free_energy]]
  - Spacetime structure
  - Causal consistency

### 2. Computational Methods
- [[neural_free_energy]]
  - Deep architectures
  - End-to-end learning
- [[symbolic_free_energy]]
  - Logical inference
  - Program synthesis

### 3. Applications
- [[robotics_applications]]
  - Planning
  - Control
- [[neuroscience_applications]]
  - Brain theory
  - Neural coding

## References
- [[friston_2010]] - "The free-energy principle: a unified brain theory?"
- [[wainwright_2008]] - "Graphical Models, Exponential Families, and Variational Inference"
- [[amari_2016]] - "Information Geometry and Its Applications"
- [[parr_2020]] - "Markov blankets, information geometry and stochastic thermodynamics"

## See Also
- [[active_inference]]
- [[variational_inference]]
- [[predictive_coding]]
- [[information_theory]]
- [[optimal_control]]
- [[belief_updating]]
- [[learning_theory]] 