---
title: Worked Example — Variational Free Energy
type: example
status: draft
created: 2025-08-08
updated: 2026-10-09
tags:
- example
- vfe
semantic_relations:
  - type: demonstrates
    links:
      - "[[variational_free_energy]]"
  - type: relates_to
    links:
      - "[[vfe_components]]"
      - "[[variational_bounds]]"
      - "[[worked_example_efe]]"
---


# Worked Example: Variational Free Energy on a Categorical Model

This page computes the variational free energy $F[q]$ for the smallest non-trivial discrete model: two hidden states, two outcomes, one observation. Every number below is reproduced by the code block at the end.

## Setup

The generative model uses the repository conventions (`A[o, s] = P(o | s)`, columns sum to one):

$$A = \begin{pmatrix} 0.9 & 0.2 \\ 0.1 & 0.8 \end{pmatrix}, \qquad p(s) = D = (0.6,\ 0.4)$$

The agent observes outcome $o = 0$. The recognition density $q(s)$ is a categorical distribution over the two states.

## Definition and decomposition

$$F[q] = \mathbb{E}_{q(s)}\big[\ln q(s) - \ln p(o, s)\big] = \underbrace{D_{\mathrm{KL}}\big(q(s)\,\|\,p(s)\big)}_{\text{complexity}} \;\underbrace{-\,\mathbb{E}_{q(s)}[\ln p(o \mid s)]}_{-\text{accuracy}}$$

A second, equivalent form shows why $F$ is a bound on surprise:

$$F[q] = D_{\mathrm{KL}}\big(q(s)\,\|\,p(s \mid o)\big) - \ln p(o)$$

## Exact quantities

The joint at the observed outcome is $p(o{=}0, s) = (0.9 \cdot 0.6,\ 0.2 \cdot 0.4) = (0.54,\ 0.08)$. The evidence is $p(o{=}0) = 0.62$, so the surprise is $-\ln 0.62 = 0.4780$ nats and the exact posterior is $p(s \mid o) = (0.8710,\ 0.1290)$.

## Free energy at four candidate beliefs

| $q(s)$ | accuracy $\mathbb{E}_q[\ln p(o\mid s)]$ | complexity | $F[q]$ |
|--------|------|------|------|
| prior $(0.6, 0.4)$ | $-0.7070$ | $0.0000$ | $0.7070$ |
| uniform $(0.5, 0.5)$ | $-0.8574$ | $0.0204$ | $0.8778$ |
| posterior $(0.871, 0.129)$ | $-0.2994$ | $0.1786$ | $0.4780$ |
| wrong $(0.2, 0.8)$ | $-1.3086$ | $0.3348$ | $1.6434$ |

Three things are visible. First, the prior has zero complexity but poor accuracy. Second, the posterior accepts a complexity cost of $0.1786$ nats to buy $0.4076$ nats of accuracy, and the trade is the best available. Third, $F$ at the posterior equals the surprise $0.4780$ exactly, because the KL term vanishes. At the uniform belief the gap is $0.8778 - 0.4780 = 0.3998$, which is $D_{\mathrm{KL}}(q \,\|\, p(s \mid o))$.

## Gradient descent on $F$

Parameterising $q = \mathrm{softmax}(\theta)$ and taking plain gradient steps of size 1 on the logits, starting from the prior, gives $F = 0.5270,\ 0.4953,\ 0.4856$ after steps 1 to 3 with beliefs $(0.755, 0.245),\ (0.805, 0.195),\ (0.828, 0.172)$. The iteration converges to the exact posterior, which is also the closed-form fixed point $q^\ast(s) \propto \exp(\ln p(o, s))$. See [[optimization_strategies]] for why a natural-gradient step reaches that fixed point in one move.

## Reproducing the numbers

```python
import numpy as np

A = np.array([[0.9, 0.2],
              [0.1, 0.8]])          # A[o, s] = P(o | s)
D = np.array([0.6, 0.4])            # prior over states
o = 0                               # observed outcome


def vfe(q):
    accuracy = np.sum(q * np.log(A[o]))
    complexity = np.sum(q * (np.log(q) - np.log(D)))
    return complexity - accuracy, accuracy, complexity


joint = A[o] * D
evidence = joint.sum()
posterior = joint / evidence
print("evidence", evidence, "surprise", -np.log(evidence))
print("posterior", posterior)
for name, q in [("prior", D), ("uniform", np.array([0.5, 0.5])),
                ("posterior", posterior), ("wrong", np.array([0.2, 0.8]))]:
    F, acc, cx = vfe(q)
    print(f"{name:9s} F={F:.4f} accuracy={acc:.4f} complexity={cx:.4f}")

# Plain gradient descent on the softmax logits of q.
ln_joint = np.log(A[o]) + np.log(D)
logits = np.log(D)
q = D.copy()
for step in range(3):
    grad_q = np.log(q) - ln_joint + 1.0
    grad_logits = q * (grad_q - np.sum(q * grad_q))
    logits = logits - grad_logits
    q = np.exp(logits) / np.exp(logits).sum()
    print("step", step + 1, q.round(4), "F =", round(vfe(q)[0], 4))
```

The repository's `DiscreteGenerativeModel.posterior(observation, prior)` in `code/tools/src/models/active_inference/generative_model.py` returns the same exact posterior, `[0.871, 0.129]`, for `observation=0` and `prior=D`. Note that `ActiveInferenceModel.calculate_free_energy` in `base.py` is a simpler proxy (half the squared prediction error plus a KL term to the prior) and does not equal the $F$ computed here.

## Related

- [[variational_free_energy]], [[vfe_components]], [[variational_bounds]], [[belief_updating]]
- [[worked_example_efe]], [[worked_examples_index]]
