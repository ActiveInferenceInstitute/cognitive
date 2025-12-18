---
title: Core Free Energy Principle
type: mathematical_concept
id: fep_core_principle_001
created: 2025-12-18
updated: 2025-12-18
tags:
  - free_energy_principle
  - mathematics
  - variational_inference
  - information_theory
  - thermodynamics
  - statistical_physics
aliases: [fep_core, variational_free_energy, free_energy_minimization]
semantic_relations:
  - type: foundation
    links:
      - [[../mathematics/variational_free_energy]]
      - [[../mathematics/expected_free_energy]]
      - [[../mathematics/markov_blankets]]
      - [[../mathematics/information_geometry]]
  - type: implements
    links:
      - [[../cognitive/perception]]
      - [[../cognitive/active_inference]]
      - [[../biology/homeostasis]]
      - [[../systems/self_organization]]
  - type: relates
    links:
      - [[../mathematics/thermodynamics]]
      - [[../mathematics/statistical_physics]]
      - [[../mathematics/information_theory]]
      - [[../mathematics/bayesian_inference]]
---

# Core Free Energy Principle

The Free Energy Principle (FEP) provides a unified mathematical framework for understanding adaptive behavior in biological and artificial systems. It states that all self-organizing systems that resist the natural tendency toward disorder must minimize their variational free energy, a statistical quantity that bounds the surprise associated with sensory signals.

## 🎯 Fundamental Statement

**The Free Energy Principle**: Any adaptive system that maintains its organization over time must minimize variational free energy.

This principle has profound implications across multiple domains:
- **Physics**: Connects to non-equilibrium thermodynamics
- **Biology**: Explains homeostasis and adaptation
- **Neuroscience**: Provides foundation for predictive processing
- **Artificial Intelligence**: Enables autonomous learning systems

## 📐 Mathematical Formulation

### Variational Free Energy

The core mathematical object of the FEP is variational free energy $F[q]$:

```math
F[q] = \mathbb{E}_q[\ln q(s) - \ln p(s,o)] = D_{KL}[q(s)||p(s|o)] - \ln p(o)
```

**Components**:
- $q(s)$: Variational posterior density over hidden states $s$
- $p(s,o)$: Generative model relating hidden states to observations $o$
- $D_{KL}$: Kullback-Leibler divergence
- $p(o)$: Marginal likelihood (evidence)

### Generative Model Structure

The FEP assumes systems embody a generative model with three key components:

```math
\begin{aligned}
p(s,o) &= p(o|s)p(s) \\
p(s) &= \prod_{t=1}^T p(s_t|s_{t-1},u_t) \\
p(o|s) &= \prod_{t=1}^T p(o_t|s_t)
\end{aligned}
```

Where:
- $s_t$: Hidden states at time $t$
- $o_t$: Observations at time $t$
- $u_t$: Control signals (actions)
- $p(o|s)$: Likelihood mapping (A matrix)
- $p(s'|s,u)$: Transition mapping (B matrix)
- $p(s)$: Prior over initial states (D vector)

### Markov Blanket Partition

Systems are separated from their environment by a Markov blanket:

```math
\begin{aligned}
\eta &= \text{external states} \\
s &= \text{sensory states} \\
a &= \text{active states} \\
\mu &= \text{internal states}
\end{aligned}
```

**Conditional Independence**: Internal states are independent of external states given the blanket:

```math
p(\mu|\eta,s,a) = p(\mu|s,a)
```

## 🔄 Variational Inference

### Perception as Inference

The FEP recasts perception as variational inference:

```math
q(s) = \arg\min_q F[q] \approx p(s|o)
```

This minimization is performed through gradient descent on the variational free energy:

```math
\frac{dq}{dt} = -\frac{\delta F}{\delta q}
```

### Fixed-Point Equations

The variational fixed points satisfy:

```math
\begin{aligned}
\ln \tilde{q}(s) &= \langle \ln p(o,s) \rangle_{q(\cdot|s)} + const \\
q(s) &= \sigma(\tilde{q}(s))
\end{aligned}
```

Where $\sigma$ is the softmax function and angle brackets denote expectations.

## 🎬 Action and Planning

### Expected Free Energy

Action selection minimizes expected free energy:

```math
G(\pi) = \mathbb{E}_{q(s|o)}[F[q] - \ln p(o|\pi)]
```

This decomposes into extrinsic and epistemic terms:

```math
G(\pi) = \mathbb{E}_{q(o,\tilde{s}|\pi)}[\ln \frac{q(o,\tilde{s}|\pi)}{p(o,\tilde{s})}] + \mathbb{E}_{q(\tilde{s}|\pi)}[D_{KL}[q(s|\tilde{s})||q(s|\tilde{s},\pi)]]
```

Where:
- First term: Extrinsic value (goal achievement)
- Second term: Epistemic value (information gain)

### Active Inference Loop

The complete active inference cycle:

1. **Perceive**: Update beliefs via variational inference
2. **Plan**: Select actions minimizing expected free energy
3. **Act**: Execute selected actions
4. **Learn**: Update generative model parameters

## 📊 Information Geometry

### Fisher Information and Natural Gradients

The FEP connects to information geometry through the Fisher information matrix:

```math
\mathcal{I}_{ij} = \mathbb{E}_q\left[\frac{\partial \ln q}{\partial \theta_i}\frac{\partial \ln q}{\partial \theta_j}\right]
```

Natural gradient descent uses this metric:

```math
\Delta\theta = -\mathcal{I}^{-1}\nabla_\theta F
```

### Statistical Manifold Structure

The space of probability distributions forms a Riemannian manifold with:
- **Metric**: Fisher information metric
- **Connections**: Exponential and mixture connections
- **Curvature**: Characterizes information flow

## ⚡ Non-Equilibrium Thermodynamics

### Entropy Production

The FEP connects to thermodynamics through entropy production:

```math
\dot{S} = \dot{S}_i + \dot{S}_e \geq 0
```

Where:
- $\dot{S}_i$: Internal entropy production
- $\dot{S}_e$: Entropy flow to environment

### Fluctuation Theorems

The principle of least action connects to fluctuation theorems:

```math
\langle e^{-W} \rangle = 1
```

Where $W$ is the work done during state transitions.

## 🧬 Biological Implementation

### Markov Blanket Dynamics

Biological systems maintain Markov blankets through coupled dynamics:

```math
\begin{aligned}
\dot{\mu} &= f_\mu(\mu,s) + \omega_\mu \\
\dot{s} &= f_s(\mu,s,a,\eta) + \omega_s \\
\dot{a} &= f_a(\mu,s,a) + \omega_a \\
\dot{\eta} &= f_\eta(s,a,\eta) + \omega_\eta
\end{aligned}
```

### Homeostatic Regulation

Homeostasis emerges from free energy minimization:

```math
F = \sum_t \frac{1}{2\sigma^2} (x_t - x^*)^2
```

Where $x^*$ represents set points and $\sigma$ represents precision.

## 🧠 Neural Implementation

### Predictive Coding

Neural predictive coding implements the FEP hierarchically:

```math
\begin{aligned}
\epsilon_t &= o_t - \hat{o}_t \\
\dot{\mu}_t &= -\frac{\partial F}{\partial \mu} + \epsilon_t \\
\hat{o}_t &= g(\mu_t)
\end{aligned}
```

### Synaptic Plasticity

Learning rules emerge from free energy minimization:

```math
\Delta w = -\eta \frac{\partial F}{\partial w}
```

This includes both Hebbian and homeostatic plasticity.

## 🔧 Implementation Framework

### Python Implementation

```python
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from typing import Dict, Tuple, List

class FreeEnergyPrinciple:
    """Core implementation of the Free Energy Principle."""

    def __init__(self,
                 n_states: int,
                 n_observations: int,
                 n_actions: int):
        """Initialize FEP system.

        Args:
            n_states: Number of hidden states
            n_observations: Number of possible observations
            n_actions: Number of possible actions
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_actions = n_actions

        # Generative model parameters
        self.A = self._init_A_matrix()  # Likelihood p(o|s)
        self.B = self._init_B_matrix()  # Transition p(s'|s,a)
        self.C = self._init_C_vector()  # Preferences p(o*)
        self.D = self._init_D_vector()  # Prior p(s)

        # Variational parameters
        self.qs = self.D.copy()  # Variational posterior

    def _init_A_matrix(self) -> jnp.ndarray:
        """Initialize likelihood matrix."""
        A = jnp.ones((self.n_observations, self.n_states))
        return A / A.sum(axis=0, keepdims=True)

    def _init_B_matrix(self) -> jnp.ndarray:
        """Initialize transition matrix."""
        B = jnp.zeros((self.n_states, self.n_states, self.n_actions))
        for a in range(self.n_actions):
            B = B.at[:, :, a].set(jnp.eye(self.n_states))
        return B

    def _init_C_vector(self) -> jnp.ndarray:
        """Initialize preference vector."""
        return jnp.ones(self.n_observations) / self.n_observations

    def _init_D_vector(self) -> jnp.ndarray:
        """Initialize prior vector."""
        return jnp.ones(self.n_states) / self.n_states

    def compute_variational_free_energy(self, observation: int) -> float:
        """Compute variational free energy F(q).

        Args:
            observation: Current observation index

        Returns:
            Variational free energy value
        """
        # Expected log likelihood
        expected_log_likelihood = jnp.sum(
            self.qs * jnp.log(self.A[observation, :] + 1e-16)
        )

        # Entropy of variational posterior
        entropy = -jnp.sum(self.qs * jnp.log(self.qs + 1e-16))

        # Expected log prior
        expected_log_prior = jnp.sum(self.qs * jnp.log(self.D + 1e-16))

        # Variational free energy
        F = expected_log_prior - expected_log_likelihood - entropy

        return float(F)

    def update_beliefs(self, observation: int, n_iterations: int = 10) -> jnp.ndarray:
        """Update variational beliefs through free energy minimization.

        Args:
            observation: Current observation
            n_iterations: Number of inference iterations

        Returns:
            Updated belief distribution
        """
        def free_energy_fn(qs):
            """Free energy as function of beliefs."""
            qs_norm = qs / jnp.sum(qs)

            expected_llh = jnp.sum(
                qs_norm * jnp.log(self.A[observation, :] + 1e-16)
            )
            entropy = -jnp.sum(qs_norm * jnp.log(qs_norm + 1e-16))
            expected_prior = jnp.sum(qs_norm * jnp.log(self.D + 1e-16))

            return expected_prior - expected_llh - entropy

        # Gradient descent on free energy
        grad_fn = grad(free_energy_fn)

        for _ in range(n_iterations):
            gradient = grad_fn(self.qs)
            self.qs = self.qs - 0.1 * gradient  # Learning rate = 0.1
            self.qs = jnp.maximum(self.qs, 1e-16)  # Positivity
            self.qs = self.qs / jnp.sum(self.qs)  # Normalization

        return self.qs

    def compute_expected_free_energy(self, action: int) -> float:
        """Compute expected free energy for action selection.

        Args:
            action: Action index

        Returns:
            Expected free energy G(π)
        """
        G = 0.0

        # Sum over possible next states and observations
        for s_next in range(self.n_states):
            transition_prob = self.B[s_next, :, action]
            qs_next = self.qs * transition_prob
            qs_next = qs_next / jnp.sum(qs_next)

            for o_next in range(self.n_observations):
                likelihood = self.A[o_next, s_next]
                qo_next = jnp.sum(qs_next * self.A[o_next, :])

                # Extrinsic value (goal achievement)
                extrinsic_value = qo_next * jnp.log(qo_next / (self.C[o_next] + 1e-16))

                # Epistemic value (information gain)
                epistemic_value = qs_next[s_next] * jnp.log(
                    qs_next[s_next] / (self.qs[s_next] + 1e-16)
                )

                G += likelihood * jnp.sum(self.qs * transition_prob) * (
                    extrinsic_value + epistemic_value
                )

        return float(G)

    def select_action(self) -> int:
        """Select action minimizing expected free energy.

        Returns:
            Selected action index
        """
        expected_free_energies = [
            self.compute_expected_free_energy(action)
            for action in range(self.n_actions)
        ]

        return int(jnp.argmin(jnp.array(expected_free_energies)))

    def perception_step(self, observation: int) -> Dict[str, jnp.ndarray]:
        """Perform perceptual inference.

        Args:
            observation: Current observation

        Returns:
            Perception results
        """
        # Update beliefs
        self.qs = self.update_beliefs(observation)

        # Compute free energy
        F = self.compute_variational_free_energy(observation)

        return {
            'beliefs': self.qs.copy(),
            'free_energy': F,
            'observation': observation
        }

    def action_step(self) -> Dict[str, float]:
        """Perform action selection.

        Returns:
            Action selection results
        """
        action = self.select_action()
        G = self.compute_expected_free_energy(action)

        expected_free_energies = [
            self.compute_expected_free_energy(a) for a in range(self.n_actions)
        ]

        return {
            'action': action,
            'expected_free_energy': G,
            'all_expected_free_energies': expected_free_energies
        }

    def learning_step(self, observation: int, action: int, next_observation: int):
        """Update generative model through learning.

        Args:
            observation: Previous observation
            action: Action taken
            next_observation: Resulting observation
        """
        # Update likelihood matrix A
        learning_rate = 0.01
        current_state = jnp.argmax(self.qs)

        self.A = self.A.at[next_observation, current_state].add(learning_rate)
        self.A = self.A / self.A.sum(axis=0, keepdims=True)

        # Update transition matrix B
        next_state = jnp.argmax(self.qs)  # Simplified
        self.B = self.B.at[next_state, current_state, action].add(learning_rate)
        self.B = self.B / self.B.sum(axis=0, keepdims=True)

    def step(self, observation: int) -> Dict:
        """Complete FEP step: perceive, act, learn.

        Args:
            observation: Current observation

        Returns:
            Complete step results
        """
        # Perceive
        perception = self.perception_step(observation)

        # Act
        action = self.action_step()

        return {
            'perception': perception,
            'action': action,
            'timestamp': getattr(self, '_timestamp', 0)
        }

# Usage example
def demo_fep():
    """Demonstrate basic FEP functionality."""
    # Initialize FEP system
    fep = FreeEnergyPrinciple(n_states=3, n_observations=3, n_actions=2)

    # Simulate observations
    observations = [0, 1, 2, 1, 0]

    for t, obs in enumerate(observations):
        print(f"\nTimestep {t}:")
        print(f"  Observation: {obs}")

        # FEP step
        result = fep.step(obs)

        print(f"  Beliefs: {result['perception']['beliefs']}")
        print(f"  Free Energy: {result['perception']['free_energy']:.3f}")
        print(f"  Selected Action: {result['action']['action']}")
        print(f"  Expected Free Energy: {result['action']['expected_free_energy']:.3f}")

if __name__ == "__main__":
    demo_fep()
```

## 🧪 Validation and Testing

### Empirical Validation

#### Free Energy Minimization Test
```python
def test_free_energy_minimization(fep_system, observations):
    """Test that free energy decreases over time."""
    free_energies = []

    for obs in observations:
        result = fep_system.step(obs)
        free_energies.append(result['perception']['free_energy'])

    # Check if free energy generally decreases
    initial_avg = np.mean(free_energies[:len(free_energies)//3])
    final_avg = np.mean(free_energies[-len(free_energies)//3:])

    assert final_avg < initial_avg, "Free energy should decrease with adaptation"
    return free_energies
```

#### Belief Accuracy Test
```python
def test_belief_accuracy(fep_system, true_states, observations):
    """Test accuracy of variational inference."""
    accuracies = []

    for true_state, obs in zip(true_states, observations):
        result = fep_system.step(obs)
        predicted_state = np.argmax(result['perception']['beliefs'])
        accuracy = 1.0 if predicted_state == true_state else 0.0
        accuracies.append(accuracy)

    return np.mean(accuracies)
```

### Theoretical Validation

#### Information-Theoretic Bounds
The FEP provides bounds on surprise:

```math
F[q] \geq -\ln p(o) \geq H[o]
```

This ensures that variational free energy provides an upper bound on the true surprise.

#### Thermodynamic Consistency
The principle maintains thermodynamic consistency through the fluctuation theorems and entropy production bounds.

## 🔗 Related Concepts

### Foundational Links
- [[../mathematics/variational_inference]] - Approximate Bayesian inference
- [[../mathematics/bayesian_inference]] - Probabilistic reasoning foundation
- [[../mathematics/information_theory]] - Information-theoretic measures
- [[../mathematics/thermodynamics]] - Physical principles

### Implementation Links
- [[../cognitive/active_inference]] - Action-oriented extension
- [[../cognitive/predictive_coding]] - Neural implementation
- [[../biology/homeostasis]] - Biological application
- [[../implementations/python_framework]] - Code implementations

### Advanced Links
- [[../mathematics/information_geometry]] - Geometric formulation
- [[../systems/self_organization]] - Complex systems perspective
- [[../philosophy/mind_body_problem]] - Philosophical implications
- [[../applications/neuroscience]] - Neural applications

## 📚 Further Reading

### Primary Literature
- Friston (2010): "The free-energy principle: a unified brain theory?"
- Parr et al. (2022): "Active inference: the free energy principle in mind, brain, and behavior"
- Friston et al. (2017): "Active Inference and Learning"

### Mathematical Foundations
- Beal (2003): "Variational algorithms for approximate Bayesian inference"
- Wainwright & Jordan (2008): "Graphical models, exponential families, and variational inference"
- Caticha (2012): "Entropic inference"

### Applications
- Buckley et al. (2017): "The free energy principle for action and perception"
- Schwartenbeck et al. (2019): "Computational mechanisms of curiosity"
- Tschantz et al. (2020): "Scaling active inference"

---

> **Core Principle**: The Free Energy Principle provides a mathematically rigorous framework for understanding how adaptive systems maintain their organization through variational free energy minimization.

---

> **Unifying Framework**: Connects thermodynamics, information theory, Bayesian inference, and self-organization into a single coherent theory of adaptive behavior.

---

> **Practical Implementation**: Enables concrete implementations in biological, artificial, and complex systems through variational inference and active inference algorithms.

---

> **Research Foundation**: Serves as the theoretical basis for active inference, predictive coding, and related approaches in cognitive science and artificial intelligence.
