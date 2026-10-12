---
title: Channel Capacity
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - information_theory
  - mutual_information
  - communication
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/information_theory]]"
      - "[[knowledge_base/mathematics/mutual_information]]"
      - "[[knowledge_base/mathematics/entropy]]"
      - "[[knowledge_base/mathematics/epistemic_value]]"
wikidata: Q870845
---

# Channel Capacity

## Overview
The capacity of a noisy channel is the largest rate, in nats or bits per use, at which information can be sent through it with arbitrarily small error. For a discrete memoryless channel it equals the maximum of the mutual information between input and output over all input distributions, and it depends only on the channel's conditional distribution.

## Definition / Formalism
For a channel $p(y\mid x)$,

$$C=\max_{p(x)} I(X;Y)=\max_{p(x)}\sum_{x,y}p(x)\,p(y\mid x)\ln\frac{p(y\mid x)}{\sum_{x'}p(x')\,p(y\mid x')} .$$

Shannon's noisy-channel coding theorem states that rates below $C$ are achievable and rates above it are not. For a binary symmetric channel with crossover probability $\epsilon$, $C=\ln2-H_b(\epsilon)$ nats, attained by a uniform input. With $\epsilon=0.1$ this is $0.3681$ nats; a brute-force search over input distributions found the same maximum at $p(x{=}0)=0.5$.

For general channels the Blahut-Arimoto algorithm iterates $p(x)\leftarrow p(x)\exp\!\big(D_{\mathrm{KL}}[p(\cdot\mid x)\,\|\,p_Y]\big)/\text{norm}$ with $p_Y=\sum_x p(x)p(\cdot\mid x)$, and converges to the capacity-achieving input. For the three-input channel with columns $(0.8,0.1,0.1)$, $(0.1,0.8,0.1)$, $(0,0,1)$ it returned $C=0.6847$ nats with input $(0.275,0.275,0.449)$, against $0.6629$ for a uniform input: the optimal input shifts weight to the noiseless symbol.

Mutual information is concave in $p(x)$, so the maximisation is a convex problem and $C$ upper-bounds $I(X;Y)$ for every input distribution. It is a best-case quantity and says nothing about any particular prior.

## Role in Active Inference
There is a concrete, checkable link. The likelihood matrix $A[o,s]=P(o\mid s)$ in this repo is exactly a discrete channel from hidden states to observations. In the expected free energy decomposition used here, $H[q(o)]=\text{ambiguity}+I(s;o)$, the information-gain term $I(s;o)$ is the mutual information of that channel under the current predictive prior over states. Hence the epistemic value any policy can obtain from one observation is bounded by $C(A)$. For $A=\begin{pmatrix}0.9&0.1\\0.1&0.9\end{pmatrix}$ that ceiling is 0.368 nats, and a belief $q(s)=(0.7,0.3)$ yields $I(s;o)=0.316$. The channel-capacity framing is not part of the repo's code; it is a way to read what the $A$ matrix allows.

## Related concepts
- [[mutual_information]] — the quantity being maximised
- [[information_theory]] — Shannon's setting
- [[entropy]] — conditional entropy defines the channel noise
- [[epistemic_value]] — information gain per observation, capped by capacity
- [[matrix_operations]] — the column-stochastic convention for $A$
- [[uncertainty_quantification]] — ambiguity as the noise part of a channel

## References
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal* 27.
- Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Wiley.
- Blahut, R. E. (1972). Computation of channel capacity and rate-distortion functions. *IEEE Transactions on Information Theory* 18, 460-473.
- Arimoto, S. (1972). An algorithm for computing the capacity of arbitrary discrete memoryless channels. *IEEE Transactions on Information Theory* 18, 14-20.
