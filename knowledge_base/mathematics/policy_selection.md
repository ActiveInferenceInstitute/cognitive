---
description: "Policy selection in Active Inference as a softmax of negative expected free energy scaled by precision, with implementation, properties and single-step to hierarchical variants."
type: mathematical_concept
id: policy_selection_001
created: 2024-02-05
modified: 2024-02-05
tags:
- mathematics
- active-inference
- policy-selection
- decision-making
aliases:
- action-selection
- policy-inference
---


# Policy Selection

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Mathematical Definition

Policy selection in Active Inference is based on the softmax of negative expected free energy:

$P(\pi) = \sigma(-\gamma G(\pi))$

where:

- $G(\pi)$ is the [[expected_free_energy]] for policy $\pi$

- $\gamma$ is the precision parameter (inverse temperature)

- $\sigma$ is the softmax function

```mermaid

graph LR

  EFE[Expected Free Energy G(π)] --> NEG[-γ·G(π)]

  NEG --> SOFT[Softmax]

  SOFT --> Q[q(π)]

  Q --> ACT[Action Selection]

```

## Components

### Expected Free Energy

- Future-oriented evaluation

- [[epistemic_value]]

- [[pragmatic_value]]

### Policy Space

- Available action sequences

- [[E_matrix]] definition

- action constraints

### Selection Mechanism

- Softmax transformation

- [[precision_parameter]]

- [[exploration_exploitation]]

 - See also: [[docs/guides/learning_paths/active_inference_security_learning_path|Security: precision bounds]]

## Implementation

```python
import numpy as np

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

# Illustrative two-state, two-outcome model with two one-step policies
A = np.array([[0.9, 0.1],                  # p(o | s)
              [0.1, 0.9]])
B = np.stack([np.eye(2),                   # action 0: stay
              np.array([[0.0, 1.0],        # action 1: flip
                        [1.0, 0.0]])])     # B[a][s', s]
C = softmax(np.array([2.0, 0.0]))          # preferred outcomes, p(o)
q_s = np.array([0.8, 0.2])                 # current beliefs over states

def expected_free_energy(a):
    """G = risk + ambiguity for a one-step policy (action a)."""
    q_s_next = B[a] @ q_s
    q_o = A @ q_s_next
    risk = np.sum(q_o * (np.log(q_o) - np.log(C)))
    ambiguity = q_s_next @ -(A * np.log(A)).sum(axis=0)
    return risk + ambiguity

G = np.array([expected_free_energy(a) for a in range(2)])
gamma = 1.0                                # precision, illustrative
print(G, softmax(-gamma * G))              # P(pi) = sigma(-gamma G(pi))
```

## Usage

Policy selection is used in:

- [[action_selection]] - Choosing actions

- [[planning]] - Multi-step planning

- [[active_inference_loop]] - Core decision step

## Properties

### Mathematical Properties

- optimality guarantees

- exploration control

- policy convergence

### Computational Properties

- sampling efficiency

- parallelization

- scalability

## Variants

### Single-Step

- greedy selection

- [[epsilon_greedy]]

- thompson sampling

### Multi-Step

- tree search

- [[monte_carlo]]

- [[trajectory_optimization]]

### Hierarchical

- [[option_framework]]

- hierarchical policies

- [[abstraction_levels]]

## Related Concepts

- [[decision_theory]]

- [[reinforcement_learning]]

- [[optimal_control]]

## Implementation Details

### Numerical Considerations

- [[temperature_annealing]]

- precision adaptation

- [[numerical_stability]]

### Optimization

- policy caching

- batch processing

- pruning strategies

## References

- friston policies - Policy theory

- [[active_inference_control]] - Control applications

- implementation examples - Code examples

## Advanced Mathematical Framework

### VFE-based Policy Selection

The connection between policy selection and VFE can be expressed through:

```math

\begin{aligned}

& \text{Discrete Time:} \\

& P(\pi) = \sigma(-\gamma G(\pi)) \text{ where } G(\pi) = \sum_\tau G_\tau(\pi) \\

& G_\tau(\pi) = D_{KL}[q(o_\tau|\pi)\,||\,p(o_\tau)] + \mathbb{E}_{q(s_\tau|\pi)}\big[H[p(o_\tau|s_\tau)]\big] \quad \text{(risk + ambiguity)} \\

& \text{Continuous Time:} \\

& P(\pi) = \sigma(-\gamma \int_t^{t+T} \mathcal{L}_\pi d\tau) \\

& \mathcal{L}_\pi = \frac{1}{2}(\dot{s} - f_\pi(s))^T\Gamma(\dot{s} - f_\pi(s)) + V_\pi(s)

\end{aligned}

```

### Planning Integration

Policy selection connects to planning through:

```math

\begin{aligned}

& \text{Value Function:} \\

& V_\pi(s_t) = -\mathbb{E}_\pi\left[\sum_{\tau=t}^T G_\tau|\pi,s_t\right] \\

& \text{Policy Gradient:} \\

& \nabla_\theta \ln P_\theta(\pi) = -\gamma\left(\nabla_\theta G(\pi) - \mathbb{E}_{P_\theta}[\nabla_\theta G]\right) \quad \text{(policy parameters } \theta\text{)} \\

& \text{Optimal Policy:} \\

& \pi^* = \argmin_\pi \mathbb{E}_\pi[G(\pi)]

\end{aligned}

```

### Hierarchical Extension

```math

\begin{aligned}

& \text{Hierarchical Policy:} \\

& P(\pi_l|s_l) = \sigma(-\gamma_l G_l(\pi_l)) \\

& G_l(\pi_l) = \sum_{\tau} F_{l,\tau}(\pi_l) + \text{KL}[q_l(s_l|\pi_l)||p_l(s_l|s_{l+1})]

\end{aligned}

```

## Implementation Framework

### Advanced Policy Selection

```python
import itertools
import numpy as np

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

# Illustrative model: two-step policies sum G over time steps
A = np.array([[0.9, 0.1], [0.1, 0.9]])               # p(o | s)
B = np.stack([np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])])   # B[a][s', s]
C = softmax(np.array([2.0, 0.0]))                    # preferred outcomes
ambiguity = -(A * np.log(A)).sum(axis=0)             # H[p(o | s)] per state

def policy_efe(policy, q_s):
    total = 0.0
    for a in policy:
        q_s = B[a] @ q_s
        q_o = A @ q_s
        total += np.sum(q_o * (np.log(q_o) - np.log(C))) + q_s @ ambiguity
    return total

policies = list(itertools.product(range(2), repeat=2))
G = np.array([policy_efe(p, np.array([0.8, 0.2])) for p in policies])
for gamma in (0.5, 4.0):                             # higher precision sharpens P(pi)
    print(gamma, softmax(-gamma * G).round(3))
```


## Advanced Policy Selection Framework

### 1. Unified Time Scale Framework

The relationship between discrete and continuous time policy selection:

```math

\begin{aligned}

& \text{Discrete Policy:} \\

& P(\pi) = \sigma(-\gamma \sum_\tau G_\tau(\pi)) \\

& \text{Continuous Policy:} \\

& P(\pi) = \sigma(-\gamma \int_t^{t+T} \mathcal{L}_\pi d\tau) \\

& \text{Bridge Equation:} \\

& \lim_{\Delta t \to 0} \sum_\tau G_\tau(\pi)\Delta t = \int_t^{t+T} \mathcal{L}_\pi d\tau

\end{aligned}

```


## Advanced Geometric Framework

### 1. Differential Geometric Structure

The geometric structure of policy spaces:

```math

\begin{aligned}

& \text{Policy Manifold:} \\

& T_\pi\mathcal{M} = \{\delta\pi : \int \delta\pi(a)da = 0\} \\

& \text{Fisher-Rao Metric:} \\

& g_{\pi}(\delta\pi_1, \delta\pi_2) = \int \frac{\delta\pi_1(a)\delta\pi_2(a)}{\pi(a)}da \\

& \text{Natural Gradient:} \\

& \nabla_{\text{nat}}G(\pi) = g_\pi^{-1}\nabla G(\pi)

\end{aligned}

```


## Field Theoretic Extensions

### 1. Policy Field Theory

The field theoretic formulation:

```math

\begin{aligned}

& \text{Policy Action:} \\

& S[\pi] = \int d^4x \sqrt{-g}\left(\frac{1}{2}g^{\mu\nu}\partial_\mu\pi\partial_\nu\pi + V(\pi)\right) \\

& \text{Field Equations:} \\

& \Box\pi + \frac{\partial V}{\partial\pi} = 0 \\

& \text{Noether Current:} \\

& J^\mu = T^{\mu\nu}\partial_\nu\pi

\end{aligned}

```


## Theoretical Considerations

### 1. Conservation Laws

```math

\begin{aligned}

& \text{Energy Conservation:} \\

& \frac{d}{dt}H(\pi,p) = 0 \\

& \text{Momentum Conservation:} \\

& \frac{d}{dt}P(\pi) = 0 \\

& \text{Angular Momentum:} \\

& \frac{d}{dt}L(\pi) = 0

\end{aligned}

```

### 2. Symmetry Principles

```math

\begin{aligned}

& \text{Gauge Symmetry:} \\

& \pi \to e^{i\alpha(x)}\pi \\

& \text{Scale Invariance:} \\

& \pi(x) \to \lambda^{\Delta}\pi(\lambda x) \\

& \text{Conformal Symmetry:} \\

& x^\mu \to f^\mu(x), \pi \to |\frac{\partial f}{\partial x}|^{-\Delta/d}\pi(f(x))

\end{aligned}

```

## Implementation Considerations

### 1. Numerical Methods

- Symplectic integration schemes

- Adaptive step size control

- Geometric integrators

### 2. Computational Efficiency

- Parallel policy evaluation

- GPU acceleration for field computations

- Sparse matrix methods

### 3. Stability Analysis

- Lyapunov functions

- Convergence guarantees

- Error bounds

## References

- amari 2022 - "Information Geometry and Its Applications"

- marsden 2022 - "Geometric Methods in Policy Optimization"

- weinberg 2022 - "Quantum Theory of Policy Fields"

- jordan 2022 - "Hamiltonian Monte Carlo in Practice"

## See Also

- [[information_geometry]]

- [[symplectic_geometry]]

- [[quantum_field_theory]]

- [[optimal_transport]]

- [[geometric_control_theory]]

## See also

- [[knowledge_base/cognitive/policy_selection|Policy Selection]]
