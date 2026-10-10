---
title: Variational Bounds and ELBO
description: "Derivation of the evidence lower bound and variational free energy as a bound on log evidence, its equivalent forms, a two-state numerical example, and tighter multi-sample bounds."
type: mathematical_concept
status: stable
created: 2025-08-08
updated: 2026-10-09
tags:
- variational-inference
semantic_relations:
  - type: relates
    links:
      - "[[variational_inference]]"
      - "[[variational_free_energy]]"
  - type: used_by
    links:
      - "[[message_passing]]"
      - "[[belief_updating]]"
  - type: relates_to
    links:
      - "[[kl_divergence]]"
      - "[[worked_example_vfe]]"
      - "[[bethe_free_energy]]"
---


# Variational Bounds and the ELBO

Exact Bayesian inference needs the model evidence $p(o) = \sum_s p(o, s)$, which is intractable for large or continuous state spaces. A variational bound replaces $\ln p(o)$ by a quantity that is cheap to compute and never exceeds it. Active Inference minimises the negative of this bound, the variational free energy $F[q]$.

## The evidence lower bound

For any distribution $q(s)$ with support covering that of $p(s \mid o)$, Jensen's inequality applied to the concave logarithm gives

$$\ln p(o) = \ln \mathbb{E}_{q(s)}\!\left[\frac{p(o, s)}{q(s)}\right] \;\ge\; \mathbb{E}_{q(s)}\big[\ln p(o, s)\big] - \mathbb{E}_{q(s)}\big[\ln q(s)\big] \equiv -F[q]$$

The right-hand side is the evidence lower bound (ELBO). The slack is not an approximation error of the bound; it is a known, non-negative quantity:

$$\ln p(o) = -F[q] + D_{\mathrm{KL}}\big(q(s)\,\|\,p(s \mid o)\big)$$

The bound is therefore tight exactly when $q(s) = p(s \mid o)$, and minimising $F$ over $q$ is the same as minimising the KL divergence to the posterior without ever evaluating $p(o)$.

## Equivalent forms

$$F[q] = \underbrace{D_{\mathrm{KL}}\big(q(s)\,\|\,p(s)\big)}_{\text{complexity}} - \underbrace{\mathbb{E}_{q}[\ln p(o \mid s)]}_{\text{accuracy}} = \underbrace{\mathbb{E}_{q}[-\ln p(o, s)]}_{\text{energy}} - \underbrace{H[q]}_{\text{entropy}}$$

The accuracy and complexity form is used in [[vfe_components]]. The energy and entropy form is the one that matches statistical physics (see [[applications_of_free_energy]]).

## Numbers from a two-state model

With $A = \begin{pmatrix}0.9 & 0.2\\0.1 & 0.8\end{pmatrix}$, prior $(0.6, 0.4)$ and observation $o = 0$ (the model of [[worked_example_vfe]]), the evidence is $p(o) = 0.62$ and $\ln p(o) = -0.4780$. For the uniform belief $q = (0.5, 0.5)$ the ELBO is $-0.8778$, so the gap is $0.3998 = D_{\mathrm{KL}}(q\,\|\,p(s\mid o))$. At the posterior $(0.871, 0.129)$ the gap is zero.

## When the bound cannot be tight

If the variational family does not contain the posterior, the minimum of $F$ over the family stays above $-\ln p(o)$. The standard case is a mean-field family $q(s_1, s_2) = q(s_1)\,q(s_2)$ facing a correlated posterior. Take the joint table $p(o, s_1, s_2) = \begin{pmatrix}0.40 & 0.05\\0.05 & 0.40\end{pmatrix}$ at the observed $o$, so $p(o) = 0.9$. The exact posterior puts $0.444$ on each diagonal cell. The best factorised $q$ is asymmetric, $q(s_1) \approx q(s_2) \approx (0.667, 0.333)$ (or its mirror image), and still leaves a gap of $0.4621$ nats; the symmetric stationary point $q = (0.5, 0.5)$ for both factors is a stationary point but not the minimum, with gap $0.4644$. Mean-field is mode-seeking: it commits to one diagonal mode. Structured families, such as the Bethe approximation behind [[belief_propagation]] and [[bethe_free_energy]], trade this loss against computation.

## Tighter multi-sample bounds

Drawing $K$ samples and averaging the importance weights inside the logarithm gives $\mathcal{L}_K = \mathbb{E}\big[\ln \tfrac1K \sum_k p(o, s_k)/q(s_k)\big]$. This bound is non-decreasing in $K$ and converges to $\ln p(o)$. For the uniform $q$ above, exact enumeration gives $\mathcal{L}_1 = -0.8778$, $\mathcal{L}_2 = -0.6779$, $\mathcal{L}_4 = -0.5650$, approaching $-0.4780$.

## Reproducing the numbers

```python
import itertools

import numpy as np

A = np.array([[0.9, 0.2], [0.1, 0.8]])
D = np.array([0.6, 0.4])
joint = A[0] * D                       # p(o=0, s)
log_evidence = np.log(joint.sum())
q = np.array([0.5, 0.5])
elbo = np.sum(q * (np.log(joint) - np.log(q)))
kl = np.sum(q * np.log(q / (joint / joint.sum())))
print(log_evidence, elbo, kl, np.isclose(log_evidence, elbo + kl))

weights = joint / q
for K in (1, 2, 4):
    bound = sum(
        np.prod(q[list(idx)]) * np.log(np.mean(weights[list(idx)]))
        for idx in itertools.product(range(2), repeat=K)
    )
    print("K =", K, round(bound, 4))

# Mean-field family against a correlated posterior.
J = np.array([[0.40, 0.05], [0.05, 0.40]])
ln_j = np.log(J)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


q1, q2 = np.array([0.9, 0.1]), np.array([0.9, 0.1])
for _ in range(50):
    q1 = softmax(ln_j @ q2)
    q2 = softmax(q1 @ ln_j)
qq = np.outer(q1, q2)
print(q1.round(3), "gap", round(float(np.sum(qq * (np.log(qq) - ln_j)) + np.log(J.sum())), 4))
```

In the repository, `DiscreteGenerativeModel.posterior` computes the exact posterior for the single-factor case, so $F$ is evaluated without sampling; the three `ActiveInferenceDispatcher` belief updates (`variational`, `mean_field`, `sampling`) all start from that exact posterior and differ in how far, and how stochastically, they move the current belief toward it.

## See also

- [[variational_inference]] · [[variational_free_energy]] · [[kl_divergence]]
- [[message_passing]] · [[belief_updating]]
