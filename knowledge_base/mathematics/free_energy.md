---
type: mathematical_concept
id: free_energy_001
created: 2024-02-05
modified: 2024-03-15
tags:
- mathematics
- active-inference
- free-energy
- variational-inference
- optimization
aliases:
- variational-free-energy
- VFE
- evidence-lower-bound
- ELBO
complexity: advanced
processing_priority: 1
semantic_relations:
  - type: implements
    links:
      - "[[active_inference]]"
      - "[[variational_inference]]"
  - type: mathematical_basis
    links:
      - "[[information_theory]]"
      - "[[probability_theory]]"
      - "[[optimization_theory]]"
  - type: relates
    links:
      - "[[expected_free_energy]]"
      - "[[path_integral_free_energy]]"
      - "[[predictive_coding]]"
---


# Free Energy Computation

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## What Makes Something a Free Energy?

At its core, a free energy is a functional (a function of functions) that measures the "energetic cost" of the mismatch between two probability distributions - typically between an approximate posterior distribution and the true distribution we're trying to model. The term "free energy" draws inspiration from statistical physics, where it represents the energy available to do useful work in a system.

Key characteristics that define a free energy functional:

1. Variational Form

   - Always involves an expectation over a variational distribution

   - Contains terms measuring both accuracy and complexity

   - Provides a tractable bound on an intractable quantity

1. Information-Theoretic Properties

   - Related to KL divergences between distributions

   - Measures information content and uncertainty

   - Balances model fit against model complexity

1. Optimization Characteristics

   - Serves as an objective function for inference

   - Has well-defined gradients

   - Minimization improves model fit

1. Thermodynamic Analogies

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

```python
import numpy as np

# Illustrative parameters: two hidden states, two possible observations
prior = np.array([0.6, 0.4])        # P(s)
A = np.array([[0.9, 0.2],           # P(o | s); each column sums to 1
              [0.1, 0.8]])
o = 0                               # observed outcome
q = np.array([0.7, 0.3])            # variational distribution Q(s)

joint = A[o] * prior                # P(o, s) at the observed o
F = np.sum(q * (np.log(q) - np.log(joint)))   # E_Q[ln Q(s) - ln P(o, s)]
```

### Alternative Formulations

#### Evidence Lower Bound (ELBO)

$F = -\text{ELBO} = -\mathbb{E}_{Q(s)}[\ln P(o|s)] + \text{KL}[Q(s)||P(s)]$

```python
# Continues the Core Definition example (same prior, A, o, q, F)
from cognitive.utils.matrix_utils import kl_divergence

complexity = kl_divergence(q, prior)          # KL[Q(s) || P(s)]
accuracy = np.sum(q * np.log(A[o]))           # E_Q[ln P(o | s)]
assert np.isclose(F, complexity - accuracy)   # F = complexity - accuracy
```

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
import numpy as np

eps = np.array([0.5, -0.2])         # prediction error (illustrative)
Pi = np.diag([4.0, 1.0])            # precision matrix (illustrative)
Sigma = np.linalg.inv(Pi)           # covariance matrix
F_pe = 0.5 * eps @ Pi @ eps + 0.5 * np.log(np.linalg.det(Sigma))  # up to a constant
```

### 2. Hierarchical Computation

The hierarchical form is evaluated level by level with the same expectation used under Core Definition.

### 3. Gradient Computation

```python
# Continues the Core Definition example (same joint, q)
def free_energy(q):
    return np.sum(q * (np.log(q) - np.log(joint)))

grad = np.log(q) - np.log(joint) + 1.0        # dF/dq(s), treating q as free
h = 1e-6
numeric = np.array([(free_energy(q + h * e) - free_energy(q - h * e)) / (2 * h)
                    for e in np.eye(len(q))])
assert np.allclose(grad, numeric, atol=1e-6)
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

- monte carlo free energy

  - Importance sampling

  - MCMC methods

- path integral methods

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

- representation learning

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

- relativistic free energy

  - Spacetime structure

  - Causal consistency

### 2. Computational Methods

- neural free energy

  - Deep architectures

  - End-to-end learning

- symbolic free energy

  - Logical inference

  - Program synthesis

### 3. Applications

- robotics applications

  - Planning

  - Control

- [[neuroscience_applications]]

  - Brain theory

  - Neural coding

## Advanced Mathematical Framework

### Rigorous Theoretical Foundation

**Definition** (Variational Free Energy Functional): For a probability space $(\Omega, \mathcal{F}, P)$ and recognition density $q \in \mathcal{Q}$, the variational free energy functional is:

$$\mathcal{F}: \mathcal{Q} \to \mathbb{R}, \quad \mathcal{F}[q] = \int_\Omega q(\omega) \ln \frac{q(\omega)}{p(\omega)} d\mu(\omega)$$

**Theorem** (Fundamental Variational Principle): The free energy functional satisfies:

1. **Non-negativity**: $\mathcal{F}[q] \geq 0$ with equality iff $q = p$

1. **Convexity**: $\mathcal{F}$ is strictly convex in $q$

1. **Lower Bound**: $\mathcal{F}[q] \geq -\ln Z$ where $Z$ is the partition function

**Proof**: By Jensen's inequality applied to the convex function $x \ln x$:

$$\mathcal{F}[q] = \mathbb{E}_q\left[\ln \frac{q}{p}\right] = -\mathbb{E}_q[\ln p] - H[q] \geq -\ln Z$$

### Stochastic Thermodynamics Integration

**Definition** (Thermodynamic Free Energy): In the context of stochastic thermodynamics:

$$F_{\text{thermo}} = \langle E \rangle - T S = \int E(x) \rho(x) dx + T \int \rho(x) \ln \rho(x) dx$$

where $\rho(x)$ is the non-equilibrium probability density and $E(x)$ is the energy function.

### Advanced Mathematical Framework

**Definition** (Free Energy Functional Space): Let $\mathcal{F}$ be the space of free energy functionals on probability distributions:

$$\mathcal{F} = \{F : \mathcal{P}(\mathcal{X}) \to \mathbb{R} \mid F \text{ is convex and lower semicontinuous}\}$$

where $\mathcal{P}(\mathcal{X})$ is the space of probability measures on state space $\mathcal{X}$.

**Theorem** (Variational Principle for Free Energy): For any generative model $P(o,s)$ and variational distribution $Q(s)$:

$$F[Q] = \mathbb{E}_Q[\ln Q(s) - \ln P(o,s)] \geq -\ln P(o)$$

with equality if and only if $Q(s) = P(s|o)$.

```python
# Continues the Core Definition example (same joint, q, F)
from cognitive.utils.matrix_utils import kl_divergence

evidence = joint.sum()              # P(o) = sum_s P(o, s)
posterior = joint / evidence        # P(s | o)
assert F >= -np.log(evidence)       # F upper-bounds -ln P(o)

# The gap is KL[Q(s) || P(s | o)]; it vanishes when Q is the exact posterior.
assert np.isclose(F + np.log(evidence), kl_divergence(q, posterior))
F_exact = np.sum(posterior * (np.log(posterior) - np.log(joint)))
assert np.isclose(F_exact, -np.log(evidence))
```

### Stochastic Calculus Extensions

**Definition** (Stochastic Free Energy): For systems with stochastic dynamics:

$$dF_t = \frac{\partial F}{\partial t} dt + \nabla F \cdot dW_t + \frac{1}{2} \text{Tr}(\nabla^2 F \cdot \Sigma) dt$$

where $dW_t$ is a Wiener process and $\Sigma$ is the diffusion matrix.

## References

- [[friston_2010]] - "The free-energy principle: a unified brain theory?"

- wainwright 2008 - "Graphical Models, Exponential Families, and Variational Inference"

- amari 2016 - "Information Geometry and Its Applications"

- parr 2020 - "Markov blankets, information geometry and stochastic thermodynamics"

- rockafellar 1970 - "Convex Analysis"

- evans 2010 - "Partial Differential Equations"

- oksendal 2003 - "Stochastic Differential Equations"

- ambrosio 2008 - "Gradient Flows in Metric Spaces"

## See Also

- [[active_inference]]

- [[variational_inference]]

- [[predictive_coding]]

- [[information_theory]]

- [[optimal_control]]

- [[belief_updating]]

- [[learning_theory]]

