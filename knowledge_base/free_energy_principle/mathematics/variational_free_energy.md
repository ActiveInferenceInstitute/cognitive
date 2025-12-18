---
title: Variational Free Energy
type: mathematical_concept
id: fep_variational_free_energy_001
created: 2025-12-18
updated: 2025-12-18
tags:
  - variational_free_energy
  - free_energy_principle
  - variational_inference
  - information_theory
  - optimization
aliases: [variational_free_energy, variational_f, free_energy_variational]
semantic_relations:
  - type: foundation
    links:
      - [[core_principle]]
      - [[../mathematics/variational_inference]]
      - [[../mathematics/kullback_leibler_divergence]]
      - [[../mathematics/entropy]]
  - type: implements
    links:
      - [[../cognitive/perception]]
      - [[../cognitive/predictive_coding]]
      - [[../biology/neural_systems]]
      - [[../implementations/python_framework]]
  - type: relates
    links:
      - [[expected_free_energy]]
      - [[../mathematics/information_geometry]]
      - [[../mathematics/path_integral]]
      - [[../mathematics/thermodynamics]]
---

# Variational Free Energy

Variational free energy is the core mathematical quantity minimized by adaptive systems according to the Free Energy Principle. It provides an upper bound on the surprise (negative log probability) of sensory signals and serves as the objective function for both perception and learning in biological and artificial systems.

## 🎯 Definition and Interpretation

### Formal Definition

Variational free energy $F[q]$ is defined as:

```math
F[q] = \mathbb{E}_q[\ln q(s) - \ln p(s,o)] = D_{KL}[q(s)||p(s|o)] - \ln p(o)
```

**Components**:
- $q(s)$: Variational posterior density over hidden states
- $p(s,o)$: Joint generative model
- $p(s|o)$: True posterior distribution
- $p(o)$: Marginal likelihood (evidence)
- $D_{KL}$: Kullback-Leibler divergence

### Information-Theoretic Interpretation

Variational free energy decomposes into accuracy and complexity terms:

```math
F[q] = \underbrace{\mathbb{E}_q[-\ln p(o|s)]}_{\text{Accuracy}} + \underbrace{D_{KL}[q(s)||p(s)]}_{\text{Complexity}}
```

- **Accuracy**: Expected negative log-likelihood (goodness of fit)
- **Complexity**: KL divergence between variational posterior and prior (regularization)

This decomposition reveals the fundamental trade-off between model fit and model complexity.

## 🔄 Variational Inference

### Perception as Variational Inference

The FEP recasts perception as minimizing variational free energy:

```math
q^*(s) = \arg\min_q F[q] \approx p(s|o)
```

### Gradient Flow Dynamics

Belief updating follows gradient descent on free energy:

```math
\frac{dq(s)}{dt} = -\frac{\delta F[q]}{\delta q(s)}
```

### Fixed-Point Equations

The variational fixed points satisfy:

```math
q(s) \propto \exp\left(\mathbb{E}_{q(\cdot|s)}[\ln p(s,o)]\right)
```

This is equivalent to the mean-field approximation in statistical physics.

## 📐 Mathematical Properties

### Bounds and Inequalities

Variational free energy provides an upper bound on surprise:

```math
F[q] \geq -\ln p(o)
```

With equality achieved when $q(s) = p(s|o)$.

### Convexity Properties

Under certain conditions, variational free energy is convex in the variational parameters, ensuring convergence of optimization algorithms.

### Information Geometry

The space of variational distributions forms a statistical manifold with the Fisher information metric providing the natural geometry for optimization.

## 🧬 Biological Implementation

### Neural Predictive Coding

In neural systems, variational free energy minimization is implemented through predictive coding:

```math
\begin{aligned}
\epsilon_t &= o_t - \hat{o}_t \\
\dot{\mu}_t &= -\frac{\partial F}{\partial \mu} + \epsilon_t \\
\hat{o}_t &= g(\mu_t)
\end{aligned}
```

Where $\epsilon_t$ represents prediction errors and $\mu_t$ represents beliefs.

### Hierarchical Processing

Free energy minimization occurs at multiple hierarchical levels:

```math
F = \sum_{l=1}^L F_l + \sum_{l=1}^{L-1} F_{l,l+1}
```

Where $F_l$ is free energy at level $l$ and $F_{l,l+1}$ represents inter-level message passing.

## 🔧 Computational Implementation

### Gradient-Based Optimization

```python
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from typing import Callable, Tuple

class VariationalFreeEnergy:
    """Implementation of variational free energy minimization."""

    def __init__(self,
                 log_likelihood: Callable,
                 log_prior: Callable,
                 state_dim: int):
        """Initialize variational free energy system.

        Args:
            log_likelihood: Log likelihood function ln p(o|s)
            log_prior: Log prior function ln p(s)
            state_dim: Dimension of state space
        """
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.state_dim = state_dim

    def compute_free_energy(self,
                           q_params: jnp.ndarray,
                           observation: jnp.ndarray) -> float:
        """Compute variational free energy F(q).

        Args:
            q_params: Parameters of variational distribution
            observation: Current observation

        Returns:
            Variational free energy value
        """
        # Get variational distribution
        q_dist = self._parameterize_distribution(q_params)

        # Expected log likelihood
        expected_log_likelihood = self._compute_expected_log_likelihood(
            q_dist, observation
        )

        # Entropy of variational distribution
        entropy = self._compute_entropy(q_dist)

        # Expected log prior
        expected_log_prior = self._compute_expected_log_prior(q_dist)

        # Variational free energy
        F = expected_log_prior - expected_log_likelihood - entropy

        return float(F)

    def _parameterize_distribution(self, params: jnp.ndarray) -> Callable:
        """Convert parameters to distribution function."""
        # For Gaussian distribution
        if len(params) == 2 * self.state_dim:
            mu, log_var = jnp.split(params, 2)

            def q_dist(s):
                var = jnp.exp(log_var)
                return jnp.exp(-0.5 * jnp.sum((s - mu)**2 / var) -
                              0.5 * jnp.sum(log_var) - 0.5 * self.state_dim * jnp.log(2*jnp.pi))
        else:
            # For categorical distribution
            def q_dist(s):
                logits = params
                return jnp.exp(logits) / jnp.sum(jnp.exp(logits))

        return q_dist

    def _compute_expected_log_likelihood(self,
                                       q_dist: Callable,
                                       observation: jnp.ndarray) -> float:
        """Compute E_q[ln p(o|s)]."""
        # Monte Carlo estimation
        n_samples = 100
        samples = self._sample_from_q(q_dist, n_samples)

        log_likelihoods = []
        for sample in samples:
            llh = self.log_likelihood(observation, sample)
            log_likelihoods.append(llh)

        return float(jnp.mean(jnp.array(log_likelihoods)))

    def _compute_entropy(self, q_dist: Callable) -> float:
        """Compute entropy H[q]."""
        # For analytical distributions, use closed form
        # For now, use Monte Carlo estimation
        n_samples = 1000
        samples = self._sample_from_q(q_dist, n_samples)

        log_probs = []
        for sample in samples:
            log_prob = jnp.log(q_dist(sample) + 1e-16)
            log_probs.append(log_prob)

        return -float(jnp.mean(jnp.array(log_probs)))

    def _compute_expected_log_prior(self, q_dist: Callable) -> float:
        """Compute E_q[ln p(s)]."""
        n_samples = 100
        samples = self._sample_from_q(q_dist, n_samples)

        log_priors = []
        for sample in samples:
            lp = self.log_prior(sample)
            log_priors.append(lp)

        return float(jnp.mean(jnp.array(log_priors)))

    def _sample_from_q(self, q_dist: Callable, n_samples: int) -> jnp.ndarray:
        """Sample from variational distribution."""
        # Simplified sampling - implement proper sampling for your distribution
        samples = []
        for _ in range(n_samples):
            # Uniform sampling in state space (simplified)
            sample = jnp.array(np.random.randn(self.state_dim))
            samples.append(sample)
        return jnp.stack(samples)

    def minimize_free_energy(self,
                           initial_params: jnp.ndarray,
                           observation: jnp.ndarray,
                           n_iterations: int = 100,
                           learning_rate: float = 0.01) -> jnp.ndarray:
        """Minimize variational free energy.

        Args:
            initial_params: Initial variational parameters
            observation: Current observation
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for gradient descent

        Returns:
            Optimized variational parameters
        """
        params = initial_params.copy()

        def objective(params):
            return self.compute_free_energy(params, observation)

        grad_fn = grad(objective)

        for _ in range(n_iterations):
            gradient = grad_fn(params)
            params = params - learning_rate * gradient

        return params

    def natural_gradient_minimization(self,
                                    initial_params: jnp.ndarray,
                                    observation: jnp.ndarray,
                                    n_iterations: int = 100,
                                    learning_rate: float = 0.01) -> jnp.ndarray:
        """Minimize free energy using natural gradients.

        Args:
            initial_params: Initial variational parameters
            observation: Current observation
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for natural gradient descent

        Returns:
            Optimized variational parameters
        """
        params = initial_params.copy()

        for _ in range(n_iterations):
            # Compute Fisher information matrix
            fisher_info = self._compute_fisher_information(params)

            # Compute ordinary gradient
            def objective(p):
                return self.compute_free_energy(p, observation)

            grad_fn = grad(objective)
            gradient = grad_fn(params)

            # Natural gradient step
            try:
                natural_gradient = jnp.linalg.solve(fisher_info, gradient)
                params = params - learning_rate * natural_gradient
            except:
                # Fallback to ordinary gradient if Fisher matrix is singular
                params = params - learning_rate * gradient

        return params

    def _compute_fisher_information(self, params: jnp.ndarray) -> jnp.ndarray:
        """Compute Fisher information matrix."""
        # Simplified Fisher information computation
        # In practice, this would require computing second derivatives
        # or using Monte Carlo estimation

        eps = 1e-5
        n_params = len(params)
        fisher = jnp.zeros((n_params, n_params))

        for i in range(n_params):
            for j in range(n_params):
                # Central difference approximation
                params_plus_i = params.at[i].add(eps)
                params_minus_i = params.at[i].add(-eps)

                if i == j:
                    # Diagonal elements
                    grad_plus = grad(lambda p: self.compute_free_energy(p, jnp.zeros(self.state_dim)))(params_plus_i)
                    grad_minus = grad(lambda p: self.compute_free_energy(p, jnp.zeros(self.state_dim)))(params_minus_i)
                    fisher = fisher.at[i, j].set((grad_plus[i] - grad_minus[i]) / (2 * eps))
                else:
                    # Off-diagonal elements
                    params_plus_both = params.at[i].add(eps).at[j].add(eps)
                    params_minus_i = params.at[i].add(-eps)
                    params_minus_j = params.at[j].add(-eps)
                    params_minus_both = params.at[i].add(-eps).at[j].add(-eps)

                    term1 = grad(lambda p: self.compute_free_energy(p, jnp.zeros(self.state_dim)))(params_plus_both)[i]
                    term2 = grad(lambda p: self.compute_free_energy(p, jnp.zeros(self.state_dim)))(params_minus_i)[i]
                    term3 = grad(lambda p: self.compute_free_energy(p, jnp.zeros(self.state_dim)))(params_minus_j)[i]
                    term4 = grad(lambda p: self.compute_free_energy(p, jnp.zeros(self.state_dim)))(params_minus_both)[i]

                    fisher = fisher.at[i, j].set((term1 - term2 - term3 + term4) / (4 * eps**2))

        return fisher

# Usage example
def demo_variational_free_energy():
    """Demonstrate variational free energy minimization."""

    # Define simple generative model
    def log_likelihood(observation, state):
        # Gaussian likelihood
        return -0.5 * jnp.sum((observation - state)**2) - 0.5 * len(observation) * jnp.log(2*jnp.pi)

    def log_prior(state):
        # Standard Gaussian prior
        return -0.5 * jnp.sum(state**2) - 0.5 * len(state) * jnp.log(2*jnp.pi)

    # Initialize variational free energy system
    vfe = VariationalFreeEnergy(log_likelihood, log_prior, state_dim=2)

    # Generate observation
    true_state = jnp.array([1.0, -0.5])
    observation = true_state + 0.1 * jnp.array(np.random.randn(2))
    observation = jnp.array(observation)

    # Initial variational parameters (mean and log variance for 2D Gaussian)
    initial_params = jnp.concatenate([jnp.zeros(2), jnp.zeros(2)])

    print("Initial parameters:", initial_params)
    print("Observation:", observation)

    # Minimize variational free energy
    optimized_params = vfe.minimize_free_energy(
        initial_params, observation, n_iterations=50
    )

    print("Optimized parameters:", optimized_params)

    # Extract optimized mean
    optimized_mean = optimized_params[:2]
    print("Optimized mean:", optimized_mean)
    print("True state:", true_state)
    print("Estimation error:", jnp.linalg.norm(optimized_mean - true_state))

if __name__ == "__main__":
    demo_variational_free_energy()
```

### Advanced Optimization Methods

#### Conjugate Gradients
```python
def conjugate_gradient_minimization(self,
                                  initial_params: jnp.ndarray,
                                  observation: jnp.ndarray,
                                  n_iterations: int = 100) -> jnp.ndarray:
    """Minimize free energy using conjugate gradients."""
    # Implementation of conjugate gradient method
    # More efficient for large parameter spaces
    pass
```

#### Quasi-Newton Methods
```python
def quasi_newton_minimization(self,
                             initial_params: jnp.ndarray,
                             observation: jnp.ndarray,
                             n_iterations: int = 100) -> jnp.ndarray:
    """Minimize free energy using quasi-Newton methods (L-BFGS)."""
    # Implementation using L-BFGS or similar
    # Builds approximation to Hessian for faster convergence
    pass
```

## 🧪 Validation and Benchmarks

### Convergence Testing

```python
def test_convergence(vfe_system, observation, initial_params, tolerance=1e-6):
    """Test convergence of variational free energy minimization."""
    params = initial_params.copy()
    free_energies = []

    for i in range(1000):
        F = vfe_system.compute_free_energy(params, observation)
        free_energies.append(F)

        # Check for convergence
        if len(free_energies) > 10:
            recent_change = abs(free_energies[-1] - free_energies[-10])
            if recent_change < tolerance:
                print(f"Converged after {i} iterations")
                break

        # Update parameters (simplified gradient descent)
        grad_F = grad(lambda p: vfe_system.compute_free_energy(p, observation))(params)
        params = params - 0.01 * grad_F

    return free_energies, params
```

### Accuracy Benchmarks

```python
def benchmark_inference_accuracy(vfe_system, test_cases):
    """Benchmark accuracy of variational inference."""
    accuracies = []

    for true_state, observation in test_cases:
        # Perform inference
        initial_params = jnp.zeros(4)  # For 2D Gaussian
        optimized_params = vfe_system.minimize_free_energy(
            initial_params, observation, n_iterations=100
        )

        # Extract inferred mean
        inferred_mean = optimized_params[:2]

        # Compute accuracy
        error = jnp.linalg.norm(inferred_mean - true_state)
        accuracies.append(error)

    return accuracies
```

## 🔗 Related Concepts

### Foundational Links
- [[core_principle]] - Core FEP formulation
- [[../mathematics/variational_inference]] - General variational methods
- [[../mathematics/kullback_leibler_divergence]] - Information divergence measure
- [[../mathematics/entropy]] - Information-theoretic entropy

### Implementation Links
- [[../cognitive/predictive_coding]] - Neural implementation
- [[../implementations/python_framework]] - Code implementations
- [[expected_free_energy]] - Action selection extension
- [[../mathematics/information_geometry]] - Geometric formulation

### Advanced Links
- [[../mathematics/path_integral]] - Path integral formulations
- [[../systems/self_organization]] - Complex systems perspective
- [[../biology/neural_systems]] - Biological implementations
- [[../applications/neuroscience]] - Neural applications

## 📚 Mathematical References

### Key Papers
- Friston (2009): "The free-energy principle: a rough guide to the brain?"
- Beal (2003): "Variational algorithms for approximate Bayesian inference"
- Winn & Bishop (2005): "Variational message passing"

### Advanced Topics
- Caticha (2012): "Entropic inference and the foundations of physics"
- Friston et al. (2017): "Active Inference and Learning"
- Parr & Friston (2018): "The anatomy of inference"

### Textbooks
- Bishop (2006): "Pattern Recognition and Machine Learning"
- Murphy (2012): "Machine Learning: A Probabilistic Perspective"
- Friston (2019): "A free energy principle for a particular physics"

---

> **Core Quantity**: Variational free energy serves as the fundamental objective function minimized by all adaptive systems according to the Free Energy Principle.

---

> **Information Trade-off**: Balances accuracy (goodness of fit) against complexity (regularization) in probabilistic inference and learning.

---

> **Computational Framework**: Provides a mathematically rigorous foundation for implementing perception, learning, and action in biological and artificial systems.

---

> **Optimization Objective**: Enables gradient-based optimization methods that respect the information geometry of probability distributions.
