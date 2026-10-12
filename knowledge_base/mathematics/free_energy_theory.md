---
type: mathematical_concept
id: free_energy_theory_001
created: 2024-02-06
modified: 2024-02-06
tags:
- mathematics
- free-energy
- variational-methods
- physics
- category-theory
aliases:
- free-energy-principle
- variational-free-energy
semantic_relations:
  - type: implements
    links:
      - "[[docs/research/research_documentation_index|Research Documentation]]"
      - "[[active_inference_pomdp]]"
  - type: uses
    links:
      - "[[variational_methods]]"
      - "[[information_theory]]"
      - "[[category_theory]]"
  - type: documented_by
    links:
      - "[[docs/implementation/implementation_guides_index|Implementation Guides]]"
      - "[[docs/api/api_documentation_index|API Documentation]]"
---


# Free Energy Theory in Cognitive Modeling

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].


## Overview

Free Energy provides a unifying framework across physics, information theory, and cognitive science. This document explores various formulations of free energy and their applications in cognitive modeling.

## Physical Free Energy

### Statistical Mechanics

```python
import numpy as np

def helmholtz_free_energy(energies, temperature):
    """F = -kT ln Z over a discrete set of microstates (k = 1)."""
    beta = 1.0 / temperature
    Z = np.sum(np.exp(-beta * energies))
    return -np.log(Z) / beta

energies = np.array([0.0, 1.0, 2.0])   # illustrative microstate energies
T = 1.5                                # illustrative temperature
F = helmholtz_free_energy(energies, T)

# Check F = U - T S under the Boltzmann distribution
p = np.exp(-(energies - F) / T)
U = np.sum(p * energies)
S = -np.sum(p * np.log(p))
print(np.isclose(p.sum(), 1.0), np.isclose(F, U - T * S))
```

## Information-Theoretic Free Energy

### Variational Free Energy

```python
import numpy as np

def variational_free_energy(q_s, prior, A, o):
    """F = E_q[ln q(s) - ln p(o, s)] for a discrete model.
    A[o, s] = p(o | s), prior[s] = p(s), q_s[s] = q(s)."""
    log_joint = np.log(A[o] + 1e-16) + np.log(prior + 1e-16)
    return np.sum(q_s * (np.log(q_s + 1e-16) - log_joint))

A = np.array([[0.9, 0.2], [0.1, 0.8]])   # illustrative likelihood, columns sum to 1
prior = np.array([0.5, 0.5])
o = 0

joint = A[o] * prior
evidence = joint.sum()
posterior = joint / evidence

# F is minimised by the exact posterior, where it equals -ln p(o)
print(np.isclose(variational_free_energy(posterior, prior, A, o), -np.log(evidence)))
print(variational_free_energy(np.array([0.5, 0.5]), prior, A, o) > -np.log(evidence))
```

### Expected Free Energy

```python
import numpy as np
from cognitive.utils.matrix_utils import expected_free_energy

A = np.array([[0.9, 0.2], [0.1, 0.8]])        # p(o | s), illustrative
B = np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2)  # p(s' | s, a)
C = np.array([2.0, 0.0])                      # log-preferences over outcomes
beliefs = np.array([0.8, 0.2])

# risk + ambiguity for each action
G = [expected_free_energy(A, B, C, beliefs, a) for a in range(B.shape[2])]
print(np.round(G, 3))
```

## Implementation Considerations

### Numerical Methods

```python
import numpy as np

def log_sum_exp(x):
    """Stable ln(sum(exp(x))), the building block of ln Z and ln p(o)."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

x = np.array([1000.0, 1001.0, 1002.0])      # np.exp(x) alone would overflow
print(np.isclose(log_sum_exp(x), 1002.0 + np.log(1 + np.exp(-1) + np.exp(-2))))
```

## Documentation Links

- [[docs/research/research_documentation_index|Research Documentation]]

- [[docs/implementation/implementation_guides_index|Implementation Guides]]

- [[docs/api/api_documentation_index|API Documentation]]

- [[docs/examples/usage_examples_index|Usage Examples]]

## Advanced Theoretical Framework

### Non-Equilibrium Statistical Mechanics Integration

**Definition** (Non-Equilibrium Free Energy): For systems far from equilibrium, the free energy incorporates dissipative processes:

$$F_{\text{neq}}(t) = F_{\text{eq}}(t) + \int_0^t \sigma(s) ds$$

where $\sigma(s)$ is the entropy production rate.

**Theorem** (Jarzynski Equality for Active Inference): For a system undergoing active inference, the work performed by the agent satisfies:

$$\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$$

where $W$ is the work done and $\Delta F$ is the free energy difference.

### Path Integral Formulation

**Definition** (Path Integral Free Energy): The free energy can be expressed as a path integral:

$$F = -\frac{1}{\beta} \ln \int \mathcal{D}[x] e^{-\beta S[x]}$$

where $S[x]$ is the action functional and $\mathcal{D}[x]$ represents integration over all paths.

## References

- friston - Free Energy Principle

- parr - Active Inference

- [[amari]] - Information Geometry

- baez - Categorical Probability Theory

- [[jarzynski]] - Non-equilibrium Work Relations

- [[crooks]] - Fluctuation Theorems

- sekimoto - Stochastic Thermodynamics

