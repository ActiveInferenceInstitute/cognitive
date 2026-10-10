---
title: Information Bottleneck
description: The method of compressing a variable while preserving information about a relevant target, defined by a mutual-information trade-off, with variational forms and a formal parallel to the accuracy-complexity decomposition of free energy.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - information_theory
  - compression
  - representation_learning
semantic_relations:
  - type: relates
    links:
      - "[[mutual_information]]"
      - "[[information_theory]]"
      - "[[channel_capacity]]"
      - "[[variational_free_energy]]"
---

# Information Bottleneck

## Overview
The information bottleneck (IB), proposed by Tishby, Pereira and Bialek, formalises relevance-aware compression. Given a source $X$ and a target $Y$ that depends on $X$, find a stochastic representation $T$ of $X$ that retains as much information about $Y$ as possible while being as compressed as possible. Relevance is defined by the task through $Y$, which distinguishes IB from rate-distortion theory with a fixed distortion measure.

## Formalism
$T$ is chosen through an encoder $p(t\mid x)$ with the Markov chain $Y\leftrightarrow X\leftrightarrow T$, minimising the Lagrangian
$$\mathcal{L}=I(X;T)-\beta\,I(T;Y),\qquad \beta\ge0.$$
Small $\beta$ gives strong compression, large $\beta$ preserves prediction. The self-consistent solution has the form $p(t\mid x)\propto p(t)\exp\!\big(-\beta\,D_{\mathrm{KL}}[p(y\mid x)\Vert p(y\mid t)]\big)$, and an iterative Blahut-Arimoto-style algorithm finds it. The set of achievable pairs $(I(X;T),I(T;Y))$ forms the information curve, and the minimal sufficient statistic of $X$ for $Y$ corresponds to the limit $\beta\to\infty$.

**Variational IB.** Because mutual information is hard to compute for neural networks, Alemi and colleagues (2017) bound it, yielding the loss
$$\mathbb{E}\big[-\log q(y\mid t)\big]+\beta\,\mathrm{KL}\big(p(t\mid x)\,\Vert\,r(t)\big),$$
for a decoder $q$ and prior $r$. This is a variational autoencoder-style objective with a weighted rate term.

**Caveats.** Claims that deep networks train in distinct compression phases following IB dynamics are contested and depend on the activation functions and on how mutual information is estimated.

## Role in Active Inference
The formal parallel is exact in structure. Variational free energy splits into accuracy, $\mathbb{E}_q[\ln p(o\mid s)]$, and complexity, $\mathrm{KL}(q(s)\Vert p(s))$. In the variational IB loss the decoder term plays the role of accuracy and the KL-to-prior term the role of complexity, with $\beta$ as a weight absent from the standard free energy (it is fixed at one). So a free-energy-minimising agent is solving a compression-prediction trade-off, and the Markov blanket as a bottleneck between internal and external states is a recurring informal idea. An IB objective differs in that it has no action or preference term; expected free energy supplies those for Active Inference. Hierarchical models are sometimes read as stacked bottlenecks, and this is an interpretation.

## Related concepts
- [[mutual_information]] — the quantity traded off in the objective
- [[kl_divergence]] — appears in the optimal encoder and in the variational bound
- [[vfe_components]] — accuracy and complexity terms of free energy
- [[markov_blanket]] — boundary that acts as an informational bottleneck
- [[channel_capacity]] — rate-limiting concept for compression
- [[renormalization_group]] — coarse-graining analogy for discarding irrelevant detail

## References
- Tishby, N., Pereira, F. C. and Bialek, W. (1999). The information bottleneck method. *Proceedings of the 37th Allerton Conference on Communication, Control and Computing*.
- Alemi, A. A., Fischer, I., Dillon, J. V. and Murphy, K. (2017). Deep variational information bottleneck. *International Conference on Learning Representations*.
