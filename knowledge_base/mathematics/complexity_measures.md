---
title: Complexity Measures
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - complexity
  - information_theory
  - kolmogorov
  - mdl
semantic_relations:
  - type: foundation
    links:
      - "[[information_theory]]"
      - "[[entropy]]"
  - type: relates
    links:
      - "[[model_complexity]]"
      - "[[variational_free_energy]]"
      - "[[computational_complexity]]"
---

# Complexity Measures

## Overview
"Complexity" has no single definition. Different measures quantify different things: the randomness of a string, the amount of structure a system has, the size of a model, or the cost of computing something. Each is useful only for the question it was built for.

## Definition / Formalism
**Shannon entropy** $H[p]=-\sum_x p(x)\ln p(x)$ measures unpredictability and is maximal for uniform distributions; it is highest for random data, so it is not a measure of structure.

**Kolmogorov complexity** $K(x)$ is the length of the shortest program that outputs $x$ on a fixed universal machine. It is uncomputable and defined up to an additive constant independent of $x$ (the invariance theorem). Lempel-Ziv compression gives a computable upper-bound proxy.

**Minimum description length** (Rissanen, 1978) selects the model minimising the sum of the code length for the model and for the data given the model, trading fit against parameter cost.

**Statistical complexity** (Crutchfield and Young) is the entropy of the causal states of a process, i.e. the stored information needed for optimal prediction; it vanishes both for constant and for fully random sequences.

**Model-evidence complexity.** Variational free energy decomposes as

$$F = \underbrace{D_{\mathrm{KL}}\big[q(s)\,\|\,p(s)\big]}_{\text{complexity}} \;-\; \underbrace{\mathbb{E}_{q(s)}\big[\ln p(o\mid s)\big]}_{\text{accuracy}},$$

where complexity is the information in bits or nats that beliefs move away from the prior.

## Role in Active Inference
The decomposition above is the concrete link: minimising free energy is an accuracy-complexity trade-off, formally analogous to MDL and Bayesian model comparison, and it penalises beliefs that diverge from priors without justification in the data. Posterior-to-prior KL is also the quantity used in Bayesian model reduction. The relationship to Kolmogorov complexity is conceptual, not an identity: free-energy complexity is relative to a specific generative model and prior, while $K$ is model-independent but uncomputable. Statistical complexity and related structure measures are used descriptively, for example on neural or collective data, but they are not part of the free energy principle itself.

## Related concepts
- [[model_complexity]] — the cognitive-science treatment of the complexity term
- [[variational_free_energy]] — where complexity and accuracy appear together
- [[model_selection]] — complexity penalties in comparing models
- [[mutual_information]] — a basic structure-sensitive quantity
- [[computational_complexity]] — resource-based rather than information-based complexity
- [[kl_divergence]] — the divergence that defines complexity in free energy

## References
- Rissanen (1978), Modeling by shortest data description, Automatica 14.
- Li & Vitanyi, An Introduction to Kolmogorov Complexity and Its Applications.
- Crutchfield & Young (1989), Inferring statistical complexity, Physical Review Letters 63.
