---
type: mathematical_concept
id: variational_free_energy_001
created: 2024-03-15
modified: 2024-03-15
tags:
- free-energy
- variational-methods
- active-inference
- statistical-mechanics
- information-theory
aliases:
- VFE
- evidence-lower-bound
- ELBO
complexity: advanced
processing_priority: 1
semantic_relations:
  - type: implements
    links:
      - "[[active_inference]]"
      - "[[free_energy_principle]]"
      - "[[policy_selection]]"
  - type: mathematical_basis
    links:
      - "[[variational_calculus]]"
      - "[[information_geometry]]"
      - "[[statistical_mechanics]]"
  - type: relates
    links:
      - "[[expected_free_energy]]"
      - "[[path_integral_free_energy]]"
      - "[[belief_updating]]"
---


# Variational Free Energy

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Overview

Variational Free Energy (VFE) is a fundamental quantity in active inference that provides a measure of the discrepancy between an agent's beliefs and reality. It serves as a unified objective function for perception, learning, and action selection in both discrete and continuous time formulations.

## Mathematical Framework

### Core Definition

The Variational Free Energy is defined as:

```math

F[q] = \mathbb{E}_q[\ln q(s) - \ln p(o,s)]

```

where:

- $q(s)$ is the variational density over states

- $p(o,s)$ is the generative model

- $\mathbb{E}_q$ denotes expectation under $q$

### Discrete Time Formulation

In discrete time, VFE decomposes into accuracy and complexity in the standard way:

```math

F_t = \underbrace{-\mathbb{E}_{q(s_t)}[\ln p(o_t\mid s_t)]}_{\text{Accuracy}} + \underbrace{D_{\mathrm{KL}}\big[q(s_t)\,\|\,p(s_t)\big]}_{\text{Complexity}}

```

### Continuous Time Extension

The continuous time formulation extends VFE to:

```math

F[q] = \int_t^{t+T} \left[\mathbb{E}_q[\ln q(s(\tau)) - \ln p(o(\tau),s(\tau))] + \frac{1}{2}\mathbb{E}_q[(\dot{s} - f(s,a))^T\Gamma(\dot{s} - f(s,a))]\right] d\tau

```

## Connection to Policy Selection

### Discrete Time Policy Selection

Policy selection in discrete time uses VFE through:

```math

P(\pi) = \sigma(-\gamma G(\pi))

```

where $G(\pi)$ is the expected free energy:

```math

G(\pi) = \sum_{\tau} \mathbb{E}_{q(o_\tau,s_\tau|\pi)}[\ln q(s_\tau|\pi) - \ln p(o_\tau,s_\tau|\pi)]

```

### Continuous Time Policy Selection

In continuous time, policy selection becomes:

```math

P(\pi) = \sigma(-\gamma \int_t^{t+T} \mathcal{L}_\pi(s(\tau), \dot{s}(\tau), a(\tau)) d\tau)

```

where $\mathcal{L}_\pi$ is the policy-specific Lagrangian.

## Implementation Framework

### 1. Variational Free Energy Computer

```python
import numpy as np

D = np.array([0.6, 0.4])                      # prior p(s), illustrative
A = np.array([[0.8, 0.3],                     # p(o | s)
              [0.2, 0.7]])
o = 1
q = np.array([0.3, 0.7])                      # any variational density

definition = np.sum(q * (np.log(q) - np.log(A[o] * D)))        # E_q[ln q - ln p(o, s)]
accuracy = -np.sum(q * np.log(A[o]))
complexity = np.sum(q * (np.log(q) - np.log(D)))               # KL[q || p(s)]
print(definition, accuracy + complexity)      # the two forms agree
```

### 2. Belief Updating

```python
import numpy as np

A = np.array([[0.8, 0.3],
              [0.2, 0.7]])                    # p(o | s), illustrative
B = np.array([[0.9, 0.2],
              [0.1, 0.8]])                    # p(s' | s), one action

q = np.array([0.5, 0.5])
for o in [0, 0, 1, 0]:                        # observation sequence
    prior = B @ q                             # predict
    q = prior * A[o]                          # correct: minimiser of F is the posterior
    q /= q.sum()
    print(o, q.round(3))
```

### 3. Policy Selection

```python
import numpy as np

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

G = np.array([2.0, 1.2, 1.5])                 # expected free energy per policy, illustrative
for gamma in (0.5, 2.0, 8.0):
    print(gamma, softmax(-gamma * G).round(3))   # larger gamma concentrates on the lowest G
```

## Advanced Concepts

### 1. Hierarchical VFE

The hierarchical extension of VFE:

```math

F_{\text{hierarchical}} = \sum_{l=1}^L F_l + \text{KL}[q_l(s_l)||p_l(s_l|s_{l+1})]

```

### 2. Amortized VFE

Amortized inference learns a map from observations to variational parameters, so that minimizing VFE becomes a single forward pass (a linear map stands in for a neural network below):

```python
import numpy as np

rng = np.random.default_rng(0)

# Conjugate model (illustrative): z ~ N(0, 1), x | z ~ N(z, sigma^2)
sigma = 0.8
z = rng.normal(size=5000)
x = z + sigma * rng.normal(size=5000)

# Exact posterior mean per datapoint: x / (1 + sigma^2). Amortise it with a learned map.
w = np.linalg.lstsq(np.c_[x, np.ones_like(x)], z, rcond=None)[0]
print(w[0], 1 / (1 + sigma**2))               # learned slope vs exact posterior slope
```

### 3. Stochastic VFE

Extension to stochastic dynamics:

```math

F_{\text{stochastic}} = F + \mathbb{E}_q[\frac{1}{2}\text{tr}(D\nabla^2\ln q)]

```


## Applications

### 1. Perception

- Belief updating

- State estimation

- Parameter learning

### 2. Action

- Policy selection

- Motor control

- Decision making

### 3. Learning

- Model learning

- Skill acquisition

- Habit formation

## References

- [[friston_2010]] - "The free-energy principle: a unified brain theory?"

- [[bogacz_2017]] - "A tutorial on the free-energy framework for modelling perception and learning"

- [[buckley_2017]] - "The free energy principle for action and perception: A mathematical review"

## See Also

- [[active_inference]]

- [[expected_free_energy]]

- [[path_integral_free_energy]]

- [[belief_updating]]

- [[policy_selection]]

## Theoretical Foundations

### Connection to Free Energy Principle

The relationship between VFE and the Free Energy Principle can be expressed through:

```math

\begin{aligned}

& \text{1. Existence:} \\

& F_{\text{existence}} = \mathbb{E}_Q[\ln Q(s) - \ln P(o,s)] \geq -\ln P(o) \\

& \text{2. Boundary:} \\

& F_{\text{markov}} = \mathbb{E}_Q[\ln Q(μ,b) - \ln P(μ,b,η)] \\

& \text{3. Dynamics:} \\

& \dot{F} = -\frac{\partial F}{\partial s}^T \Gamma \frac{\partial F}{\partial s} \leq 0

\end{aligned}

```

### Information Geometric Structure

The geometry of VFE manifolds:

```math

\begin{aligned}

& \text{Fisher Metric:} \\

& g_{ij} = \mathbb{E}_Q\left[\frac{\partial \ln Q}{\partial θ_i}\frac{\partial \ln Q}{\partial θ_j}\right] \\

& \text{Natural Gradient:} \\

& \dot{θ} = -g^{-1}\frac{\partial F}{\partial θ} \\

& \text{Geodesic Flow:} \\

& \ddot{θ}^i + \Gamma^i_{jk}\dot{θ}^j\dot{θ}^k = 0

\end{aligned}

```

### Statistical Physics Connection

The thermodynamic interpretation:

```math

\begin{aligned}

& F = U - TS \\

& U = \mathbb{E}_Q[E(s)] \\

& S = -\mathbb{E}_Q[\ln Q(s)] \\

& \beta = \frac{1}{T} = \text{precision}

\end{aligned}

```


## Advanced Mathematical Bridges

### 1. Path Integral to VFE Bridge

The connection between path integral formulation and VFE:

```math

\begin{aligned}

& \text{Path Integral VFE:} \\

& F_{\text{PI}}[q] = \int \mathcal{D}[s(\tau)] q[s(\tau)] \left(\ln q[s(\tau)] - \ln p[s(\tau),o(\tau)]\right) \\

& \text{Discrete-Continuous Bridge:} \\

& F_{\text{bridge}} = \lim_{\Delta t \to 0} \sum_t F_t\Delta t = \int_0^T F(\tau)d\tau \\

& \text{Action-Value Relationship:} \\

& S[s(\tau)] = \beta \int_0^T \mathcal{L}(s,\dot{s},t)d\tau = -\ln p[s(\tau)]

\end{aligned}

```

### 2. Information Geometric Bridge

```math

\begin{aligned}

& \text{Fisher-Rao Metric:} \\

& g_{\mu\nu}(θ) = \mathbb{E}_{q_θ}\left[\frac{\partial \ln q_θ}{\partial θ^\mu}\frac{\partial \ln q_θ}{\partial θ^\nu}\right] \\

& \text{Natural Gradient Flow:} \\

& \dot{θ}^\mu = -g^{\mu\nu}(θ)\frac{\partial F}{\partial θ^\nu} \\

& \text{Wasserstein Gradient:} \\

& \nabla_W F = -\text{div}(\rho\nabla\frac{\delta F}{\delta \rho})

\end{aligned}

```

### 3. Quantum-Classical Bridge

```math

\begin{aligned}

& \text{Quantum VFE:} \\

& F_Q = \text{Tr}[\rho(\ln\rho - \ln\sigma)] + \beta\text{Tr}[\rho H] \\

& \text{Classical Limit:} \\

& \lim_{\hbar \to 0} F_Q = F_{\text{classical}} \\

& \text{Quantum Policy:} \\

& |\psi_\pi\rangle = \sum_a \sqrt{P(a|\pi)}|a\rangle

\end{aligned}

```


## Advanced Theoretical Extensions

### 1. Relativistic VFE Framework

```math

\begin{aligned}

& \text{Covariant VFE:} \\

& F_{\text{cov}} = \int d^4x \sqrt{-g} \left(T^{\mu\nu}\nabla_\mu\nabla_\nu\ln\rho + V[\rho]\right) \\

& \text{Spacetime Action:} \\

& S_{\text{spacetime}} = \int d^4x \sqrt{-g}\mathcal{L}(\phi, \partial_\mu\phi) \\

& \text{Causal Structure:} \\

& \delta F_{\text{cov}}/\delta\rho = 0 \text{ on } J^+(x)

\end{aligned}

```

### 2. Quantum Field Theory Extension

```math

\begin{aligned}

& \text{Field VFE:} \\

& F_{\text{field}} = \int \mathcal{D}[\phi] \rho[\phi] \left(\ln\rho[\phi] - \ln P[\phi]\right) \\

& \text{Effective Action:} \\

& \Gamma[\phi] = -\ln \int \mathcal{D}[\chi] \exp(-S[\chi]) \\

& \text{Ward Identity:} \\

& \frac{\delta \Gamma}{\delta \phi} = \langle\frac{\delta S}{\delta \phi}\rangle

\end{aligned}

```

### 3. Topological Extensions

```math

\begin{aligned}

& \text{Topological VFE:} \\

& F_{\text{top}} = \oint_{\partial M} \omega + \int_M d\omega \\

& \text{Characteristic Classes:} \\

& c_1(F) = \frac{i}{2\pi}\text{tr}(F) \\

& \text{Index Theorem:} \\

& \text{index}(D) = \int_M \hat{A}(M)\text{ch}(E)

\end{aligned}

```

## Implementation Considerations

### 1. Advanced Numerical Methods

- Symplectic integration for Hamiltonian dynamics

- Adaptive Runge-Kutta methods

- Stochastic variational integrators

### 2. Parallel Computing Strategies

- GPU-accelerated path integral computation

- Distributed belief propagation

- Multi-scale parallel processing

### 3. Optimization Techniques

- Natural gradient methods

- Hamiltonian Monte Carlo

- Riemannian optimization

## References

- friston 2022 - "The Free Energy Principle: A Unified Brain Theory?"

- [[parr_2022]] - "Active Inference: The Free Energy Principle in Mind, Brain, and Behavior"

- da costa 2022 - "The Mathematics of Active Inference"

- ramstead 2022 - "Neural and cognitive architectures for active inference"

## See Also

- [[quantum_field_theory]]

- [[differential_geometry]]

- [[stochastic_processes]]

- [[information_geometry]]

- [[topological_quantum_field_theory]]

## See also

- [[knowledge_base/cognitive/variational_free_energy|Variational Free Energy]]
- [[knowledge_base/free_energy_principle/mathematics/variational_free_energy|Variational Free Energy]]
