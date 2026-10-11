---
title: Worked Example — Expected Free Energy
type: example
status: draft
created: 2025-08-08
updated: 2026-10-09
tags:
- example
- efe
semantic_relations:
  - type: demonstrates
    links:
      - "[[expected_free_energy]]"
  - type: relates_to
    links:
      - "[[worked_example_vfe]]"
      - "[[policy_selection]]"
      - "[[epistemic_value]]"
---


# Worked Example: Expected Free Energy on a Toy POMDP

This page evaluates expected free energy (EFE) for a two-state, two-observation, two-action POMDP using the repository's `DiscreteGenerativeModel`, then ranks one- and two-step policies and turns them into action probabilities. All numbers are reproduced by the code block at the end.

## Convention

The repository uses the canonical decomposition

$$G(\pi,\tau) = \underbrace{D_{\mathrm{KL}}\big(q(o_\tau \mid \pi)\,\|\,p^\ast(o)\big)}_{\text{risk}} + \underbrace{\mathbb{E}_{q(s_\tau \mid \pi)}\big[H[P(o \mid s)]\big]}_{\text{ambiguity}}$$

with $p^\ast(o) = \mathrm{softmax}(C)$. Ambiguity equals $H[q(o)] - I(s;o)$, so the expected information gain is already inside $G$; the model returns it only as a diagnostic. Equivalently, $G = -I(s;o) - \mathbb{E}_{q(o)}[\ln p^\ast(o)]$, which is negative epistemic value minus pragmatic value. Adding $I(s;o)$ to $G$ a second time would weight information seeking twice (see the "Corrected mathematics" entry in `CHANGELOG.md`).

## Setup

States $\{0, 1\}$, observations $\{0, 1\}$, actions $\{\text{stay}, \text{move}\}$. State 1 is ambiguous: it produces both outcomes equally often. Outcome 0 is preferred.

$$A = \begin{pmatrix} 0.9 & 0.5 \\ 0.1 & 0.5 \end{pmatrix},\quad B_{\text{stay}} = I,\quad B_{\text{move}} = \begin{pmatrix} 0.9 & 0.9 \\ 0.1 & 0.1 \end{pmatrix},\quad C = (2, 0)$$

Preferences are $p^\ast(o) = \mathrm{softmax}(C) = (0.8808,\ 0.1192)$. Current beliefs are $q(s) = (0.3,\ 0.7)$.

## One-step EFE per action

| action | $q(s')$ | $q(o)$ | risk | ambiguity | $G$ | $I(s;o)$ |
|--------|---------|--------|------|-----------|------|---------|
| stay | $(0.3, 0.7)$ | $(0.62, 0.38)$ | $0.2229$ | $0.5827$ | $0.8056$ | $0.0813$ |
| move | $(0.9, 0.1)$ | $(0.86, 0.14)$ | $0.0020$ | $0.3619$ | $0.3639$ | $0.0431$ |

Moving wins on both terms. It pushes the agent into the unambiguous state (lower ambiguity) and makes the predicted outcomes match the preference (near-zero risk). The check $H[q(o)] - I(s;o) = 0.6641 - 0.0813 = 0.5827$ for "stay" confirms that ambiguity already contains the information gain.

## Two-step policies

Summing undiscounted one-step EFE along the predicted belief trajectory gives:

| policy | $G(\pi)$ | $P(\pi) = \sigma(-G)$ |
|--------|---------|------|
| (stay, stay) | $1.6112$ | $0.1352$ |
| (stay, move) | $1.1694$ | $0.2104$ |
| (move, stay) | $0.7277$ | $0.3272$ |
| (move, move) | $0.7277$ | $0.3272$ |

The last two tie because after one move the belief is $(0.9, 0.1)$, which is a fixed point of both actions.

## From policies to an action

`ActiveInferenceDispatcher.calculate_expected_free_energy` reduces policies to one value per first action. With the default `PolicyAggregation.MIN` the values are $(1.1694,\ 0.7277)$, and the variational policy inference applies $\mathrm{softmax}(-G/T)$ with `temperature` $T = 1$, giving $P(a) = (0.3913,\ 0.6087)$. The `MARGINAL` aggregation would instead take a soft minimum over the policies that share a first action, $-T \ln \sum_\pi \exp(-G_\pi/T)$. Note that `temperature`, not `precision_init`, scales the softmax in the variational and sampling policy paths.

## Reproducing the numbers

```python
import numpy as np

from cognitive.models.active_inference import (
    ActiveInferenceDispatcher, DiscreteGenerativeModel, InferenceConfig, ModelState)
from cognitive.utils.matrix_utils import expected_free_energy, softmax

A = np.array([[0.9, 0.5],
              [0.1, 0.5]])                  # A[o, s]; state 1 is ambiguous
B = np.zeros((2, 2, 2))
B[:, :, 0] = np.eye(2)                      # action 0: stay
B[:, :, 1] = [[0.9, 0.9], [0.1, 0.1]]       # action 1: move toward state 0
C = np.array([2.0, 0.0])                    # log preferences over observations
model = DiscreteGenerativeModel(A, B, C, np.full(2, 0.5), np.full(2, 0.5))
beliefs = np.array([0.3, 0.7])

for a in range(model.num_actions):
    G, risk, ambiguity, epistemic = model.expected_free_energy(beliefs, a)
    q_o = model.predict_observations(model.predict_states(beliefs, a))
    pragmatic = -np.sum(q_o * np.log(model.preferences))
    assert np.isclose(G, pragmatic - epistemic)          # G = -I(s;o) - E[ln p*(o)]
    assert np.isclose(G, expected_free_energy(A, B, C, beliefs, a))
    print(a, f"G={G:.4f} risk={risk:.4f} ambiguity={ambiguity:.4f} info_gain={epistemic:.4f}")

policies = model.enumerate_policies(2)
G_pi = np.array([model.evaluate_policy(beliefs, p, discount_factor=1.0) for p in policies])
for p, g, prob in zip(policies, G_pi, softmax(-G_pi)):
    print(p, f"G={g:.4f} P={prob:.4f}")

config = InferenceConfig(method="variational", policy_type="discrete", temporal_horizon=2,
                         learning_rate=0.5, precision_init=1.0, discount_factor=1.0)
dispatcher = ActiveInferenceDispatcher(config, model)
state = ModelState(beliefs=beliefs, policies=np.full(2, 0.5), precision=1.0,
                   free_energy=0.0, prediction_error=0.0)
print("first-action G", dispatcher.calculate_expected_free_energy(state))
print("P(a)", dispatcher.dispatch_policy_inference(state))
```

## Related

- [[expected_free_energy]], [[epistemic_value]], [[policy_selection]], [[mutual_information]]
- [[worked_example_vfe]], [[worked_examples_index]]

## References

- Friston, K., Rigoli, F., Ognibene, D., Mathys, C., FitzGerald, T., & Pezzulo, G. (2015). Active inference and epistemic value. Cognitive Neuroscience, 6(4), 187-214.
- Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2020). Active inference on discrete state-spaces: a synthesis. Journal of Mathematical Psychology, 99, 102447.
- Parr, T., & Friston, K. (2019). Generalised free energy and active inference. Biological Cybernetics, 113(5-6), 495-513.
- Sajid, N., Ball, P. J., Parr, T., & Friston, K. J. (2021). Active inference: demystified and compared. Neural Computation, 33(3), 674-712.
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.
