---
title: Worked Examples Index
type: index
status: stable
created: 2025-08-08
updated: 2026-10-09
tags:
- examples
- vfe
- efe
semantic_relations:
  - type: aggregates
    links:
      - "[[worked_example_vfe]]"
      - "[[worked_example_efe]]"
  - type: relates_to
    links:
      - "[[variational_bounds]]"
      - "[[mutual_information]]"
---


# Worked Examples Index

Small, fully numeric calculations that can be checked by hand and reproduced with the code in `code/tools/src`. Each page ends with a runnable Python block whose printed output matches the tables on the page.

## Pages

- [[worked_example_vfe|Variational Free Energy on a categorical model]]: one observation, two hidden states. Computes accuracy, complexity and $F[q]$ for four candidate beliefs, checks $F[q] = D_{\mathrm{KL}}(q\,\|\,p(s\mid o)) - \ln p(o)$, and runs gradient descent on the logits of $q$.
- [[worked_example_efe|Expected Free Energy on a toy POMDP]]: two states, two outcomes, two actions. Computes risk and ambiguity for each action, ranks two-step policies, and produces action probabilities through `ActiveInferenceDispatcher`.

## How the two examples connect

Both pages use the same representation. A likelihood matrix `A[o, s] = P(o | s)` has columns that sum to one; a prior vector `D` or belief vector `q(s)` sits over states; and `C` holds log preferences over observations. The VFE example is the perception step: given an outcome, find $q(s)$ that minimises $F$. The EFE example is the planning step: given $q(s)$, score each action by the free energy expected after taking it.

The two quantities share building blocks. The accuracy term of $F$ uses $\ln P(o\mid s)$, while the ambiguity term of $G$ uses its entropy $H[P(o\mid s)]$. The complexity term is a KL divergence over states, and the risk term of $G$ is a KL divergence over outcomes. A third shared object is the mutual information $I(s;o)$, which appears as the gap between the entropy of predicted outcomes and the ambiguity.

## Conventions used throughout

| Symbol | Meaning | Code |
|--------|---------|------|
| $A$ | $P(o \mid s)$, column stochastic | `DiscreteGenerativeModel.A` |
| $B$ | $P(s' \mid s, a)$, indexed `B[s_next, s_prev, a]` | `DiscreteGenerativeModel.B` |
| $C$ | log preferences; $p^\ast(o) = \mathrm{softmax}(C)$ | `DiscreteGenerativeModel.preferences` |
| $D$ | prior over initial states | `DiscreteGenerativeModel.D` |
| $G$ | expected free energy, risk plus ambiguity | `DiscreteGenerativeModel.expected_free_energy` |

Expected free energy is risk plus ambiguity throughout. The information-gain term is not added again. Log base is natural, so all quantities are in nats.

## Reproducing everything

Both scripts run from the repository root with the project virtual environment, which exposes the package as `cognitive`. This helper extracts and runs the first Python block of each page:

```python
from pathlib import Path

fence = "`" * 3
for page in ("worked_example_vfe", "worked_example_efe"):
    text = Path(f"knowledge_base/mathematics/{page}.md").read_text(encoding="utf-8")
    block = text.split(fence + "python\n", 1)[1].split(fence, 1)[0]
    print("running", page)
    exec(block)
```

## Foundations

- [[variational_free_energy]] · [[expected_free_energy]] · [[policy_selection]] · [[precision_parameter]] · [[softmax_function]] · [[numerical_stability]]
