---
title: Applications of Free Energy
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - mathematics
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/free_energy_principle/README]]"
      - "[[knowledge_base/mathematics/variational_methods]]"
      - "[[knowledge_base/mathematics/variational_free_energy]]"
      - "[[knowledge_base/mathematics/variational_bounds]]"
      - "[[knowledge_base/mathematics/thermodynamics]]"
      - "[[knowledge_base/mathematics/predictive_coding]]"
---

# Applications of Free Energy

One functional appears in several fields under different names. For a distribution $q$ over configurations $x$ and an energy $E(x)$ (or negative log joint $-\ln p(o, x)$),

$$F[q] = \mathbb{E}_q[E(x)] - T\,H[q] = -T \ln Z + T\,D_{\mathrm{KL}}\big(q\,\|\,p_{\mathrm{Gibbs}}\big), \qquad p_{\mathrm{Gibbs}}(x) = e^{-E(x)/T}/Z$$

The functional is minimised by the Gibbs distribution, and its minimum is $-T\ln Z$. Setting $T = 1$ and $E(x) = -\ln p(o, x)$ gives the variational free energy of Active Inference, with minimum $-\ln p(o)$ ([[variational_bounds]]).

## Statistical physics

In thermodynamics $F = U - TS$ is the Helmholtz free energy: mean energy minus temperature times entropy, minimised at equilibrium ([[thermodynamics]], [[statistical_physics]]). The variational statement is the Gibbs variational principle above. For three states with energies $E = (0, 1, 2)$ and $T = 1$, the partition function is $Z = 1.5032$, the minimum is $-\ln Z = -0.4076$ at $p = (0.665, 0.245, 0.090)$, and the uniform distribution has $F = -0.0986$. The difference $0.3090$ equals $T\,D_{\mathrm{KL}}(\text{uniform}\,\|\,p_{\mathrm{Gibbs}})$. The same algebra explains why the policy distribution $\mathrm{softmax}(-G/T)$ is a Boltzmann distribution with expected free energy as the energy.

## Machine learning

Maximising the evidence lower bound is minimising $F$. The recipes differ in what $q$ is. Variational Bayes and mean-field inference use a factorised $q$ updated by coordinate ascent. The expectation-maximisation algorithm alternates setting $q$ to the exact posterior (E-step, $F$ equals $-\ln p(o \mid \theta)$) and optimising model parameters $\theta$ (M-step). Variational autoencoders amortise $q$ with a neural network and optimise the bound by stochastic gradients, trading accuracy of the bound for speed ([[stochastic_optimization]]). In each case the accuracy and complexity decomposition ([[vfe_components]]) is the reconstruction versus regularisation trade-off.

## Neuroscience

Under a Gaussian generative model and a Laplace-approximate posterior, $F$ reduces to a sum of precision-weighted squared prediction errors, $\tfrac12 \Pi_o \varepsilon_o^2 + \tfrac12 \Pi_x \varepsilon_x^2$. Gradient descent on it is predictive coding ([[predictive_coding]]): error units carry $\varepsilon$ up, predictions carry $\mu$ down, and precision $\Pi$ sets the gain on each error. `ContinuousActiveInference` in `code/Things/Continuous_Generic/continuous_generic.py` implements this for generalised coordinates and descends the gradient of a free energy of exactly that form.

## Biology and control

Homeostasis is staying within viable bounds, which Active Inference expresses as a preference for observations in those bounds. `HomeostaticInference` in `code/tools/src/models/active_inference/homeostatic.py` pushes a target distribution over states through the likelihood to obtain preferred observations and chooses actions whose predicted beliefs are close to the target. Its reported `free_energy` is a proxy: half the mean squared prediction error plus $D_{\mathrm{KL}}$ from beliefs to the target. It is not the full $F$ of this page, and `ActiveInferenceModel.calculate_free_energy` in `base.py` is a proxy of the same kind. For planning, expected free energy extends the idea from the present to futures ([[expected_free_energy]]).

## Caveats

The physical and the informational readings are formally identical but not interchangeable: temperature in $F = U - TS$ is a physical quantity, whereas in softmax policy selection it is a tuning parameter ([[temperature_parameter]]). The claim that living systems minimise variational free energy is a modelling hypothesis, developed in the [[knowledge_base/free_energy_principle/README|free energy principle]] section; the calculations on this page only establish the mathematics.

## Reproducing the physics numbers

```python
import numpy as np

energies = np.array([0.0, 1.0, 2.0])
kT = 1.0
boltzmann = np.exp(-energies / kT)
Z = boltzmann.sum()
p = boltzmann / Z


def helmholtz(q):
    return float(np.sum(q * energies) + kT * np.sum(q * np.log(q)))


uniform = np.full(3, 1 / 3)
kl = float(np.sum(uniform * np.log(uniform / p)))
print(Z, -kT * np.log(Z), helmholtz(p), p.round(3))
print(helmholtz(uniform), helmholtz(uniform) + kT * np.log(Z), kT * kl)
```

## Related

- [[knowledge_base/free_energy_principle/README]]
- [[knowledge_base/mathematics/variational_methods]]
- [[variational_bounds]] · [[vfe_components]] · [[expected_free_energy]]
