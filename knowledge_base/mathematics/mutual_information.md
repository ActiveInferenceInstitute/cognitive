---
title: Mutual Information
type: mathematical_concept
status: stable
created: 2025-08-08
updated: 2026-10-09
tags:
- information-theory
semantic_relations:
  - type: relates
    links:
      - "[[entropy]]"
      - "[[kl_divergence]]"
  - type: used_by
    links:
      - "[[expected_free_energy]]"
      - "[[epistemic_value]]"
  - type: relates_to
    links:
      - "[[mutual_information_estimators]]"
      - "[[information_gain]]"
      - "[[worked_example_efe]]"
wikidata: Q252973
---


# Mutual Information

Mutual information measures how much knowing one random variable reduces uncertainty about another. In Active Inference it is the quantity that turns "look at something informative" into a term of the objective: the expected information a future observation carries about hidden states.

## Definition

For discrete variables with joint $p(x, y)$ and marginals $p(x)$, $p(y)$:

$$I(X;Y) = D_{\mathrm{KL}}\big(p(x,y)\,\|\,p(x)p(y)\big) = H[X] + H[Y] - H[X,Y] = H[X] - H[X \mid Y]$$

Properties that matter in practice: $I(X;Y) = I(Y;X)$; $I(X;Y) \ge 0$ with equality exactly when $X$ and $Y$ are independent; $I(X;Y) \le \min(H[X], H[Y])$; and the data-processing inequality $I(X;Z) \le I(X;Y)$ whenever $X \to Y \to Z$ forms a Markov chain. Units are nats with the natural logarithm, bits with $\log_2$.

## A worked value

Take the joint $p(x,y) = \begin{pmatrix}0.4 & 0.1\\0.1 & 0.4\end{pmatrix}$. Both marginals are uniform, so $H[X] = H[Y] = \ln 2 = 0.6931$, and $H[X,Y] = 1.1935$. Hence $I(X;Y) = 0.6931 + 0.6931 - 1.1935 = 0.1927$ nats, which is $0.2781$ bits. The product of marginals gives $I = 0$ by construction.

## Role in Active Inference

Let $q(s)$ be the predicted state distribution under a policy and $P(o \mid s)$ the likelihood. Define the joint $q(s, o) = P(o \mid s)\,q(s)$. Then

$$I(s;o) = H[q(o)] - \mathbb{E}_{q(s)}\big[H[P(o \mid s)]\big] = \mathbb{E}_{q(o)}\Big[D_{\mathrm{KL}}\big(q(s \mid o)\,\|\,q(s)\big)\Big]$$

The middle expression is the entropy of predicted outcomes minus the ambiguity of the likelihood. The right-hand expression is the expected information gain, the quantity called epistemic value in [[epistemic_value]]. Because the repository's expected free energy is risk plus ambiguity, and ambiguity equals $H[q(o)] - I(s;o)$, the information gain is already inside $G$; see [[expected_free_energy]].

In the two-state model of [[worked_example_efe]], with beliefs $(0.3, 0.7)$ the "stay" action has $I(s;o) = 0.0813$ nats and the "move" action has $0.0431$ nats. Stay is more informative because it leaves the agent uncertain about which state it is in, so an observation can still teach something; move collapses the belief toward a state and leaves less to learn. Information gain is largest at intermediate beliefs and is zero at a deterministic belief, which is why exploration stops by itself once uncertainty is resolved.

## Code

`DiscreteGenerativeModel.expected_free_energy` in `code/tools/src/models/active_inference/generative_model.py` returns the information gain as the fourth element of its tuple, computed as $H[q(s)]$ minus the expected posterior entropy. That is equivalent to the expression above and is clamped at zero. There is no standalone mutual-information function; for a joint table, combine the entropy helper with the KL helper:

```python
import numpy as np

from cognitive.models.matrices.matrix_ops import MatrixOps
from cognitive.utils.matrix_utils import compute_entropy

joint = np.array([[0.4, 0.1], [0.1, 0.4]])
px, py = joint.sum(axis=1), joint.sum(axis=0)

via_kl = MatrixOps.compute_kl_divergence(joint.ravel(), np.outer(px, py).ravel())
via_entropy = (
    compute_entropy(px) + compute_entropy(py) - compute_entropy(joint.ravel())
)
print(float(via_kl), float(via_entropy))  # both approximately 0.1927
```

The repository's logarithms add a floor of $10^{-12}$, so values agree with the exact result to about that precision.

## See also

- [[entropy]] · [[kl_divergence]] · [[expected_free_energy]] · [[epistemic_value]]
- [[mutual_information_estimators]] for estimating $I$ from samples
