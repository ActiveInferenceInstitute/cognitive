---
title: Information Theory Axioms
description: The axiomatic characterisations of Shannon entropy by Shannon, Khinchin and Faddeev, and the relaxations that give Renyi entropies, which underpin information gain and epistemic value.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - information_theory
  - entropy
  - foundations
semantic_relations:
  - type: relates
    links:
      - "[[information_theory]]"
      - "[[entropy]]"
      - "[[information_gain]]"
      - "[[epistemic_value]]"
---

# Information Theory Axioms

## Overview
Shannon entropy $H(p)=-\sum_i p_i\log p_i$ is not an arbitrary definition: a small set of natural requirements on an uncertainty measure forces it up to a constant factor. Several equivalent axiom systems exist, and weakening one axiom yields a family of alternative entropies. The page lists these because the information-gain and epistemic-value pages rely on the properties they guarantee.

## Characterisations
**Shannon's (1948) conditions** for a measure $H(p_1,\dots,p_n)$:
1. Continuity in the $p_i$.
2. For the uniform distribution on $n$ outcomes, $H$ increases with $n$.
3. Grouping (recursivity): if a choice is split into successive choices, the original $H$ equals the weighted sum of the component $H$ values.

The unique solution is $H=-K\sum_i p_i\log p_i$ with $K>0$ fixing the unit.

**Khinchin's (1957) conditions:** continuity; maximality, $H(p)\le H(1/n,\dots,1/n)$; expansibility, adding an outcome of probability zero leaves $H$ unchanged; and strong additivity, $H(X,Y)=H(X)+H(Y\mid X)$. These also force the Shannon form. Faddeev (1956) gave a shorter set using continuity, symmetry and a grouping rule.

**Relaxation.** Replace strong additivity by additivity only for independent variables, $H(X,Y)=H(X)+H(Y)$ when independent, and the Renyi entropies $H_\alpha=\frac1{1-\alpha}\log\sum_ip_i^\alpha$ result, with $H_1$ the Shannon limit. Non-additive proposals such as Tsallis entropy drop additivity altogether.

**Derived properties.** The axioms lead to non-negativity, $H\le\log n$, the chain rule, mutual information $I(X;Y)=H(X)-H(X\mid Y)\ge0$, and the data-processing inequality. Relative entropy $D_{\mathrm{KL}}(p\Vert q)\ge0$ with equality only at $p=q$ also has axiomatic characterisations.

## Role in Active Inference
Information gain, epistemic value and variational free energy are built from these quantities, so their properties are inherited. Expected information gain is a mutual information, non-negative by the axioms, which is why epistemic value never penalises acquiring information in expectation. Entropy-based decompositions of expected free energy, risk plus ambiguity ([[expected_free_energy]]), use the chain rule and KL positivity. The choice of Shannon over Renyi entropy is a modelling decision of the framework, not a theorem of it: Active Inference uses Shannon quantities because they match Bayesian inference and the free-energy bound, while generalised entropies have been explored as alternatives.

## Related concepts
- [[entropy]] — the quantity characterised by the axioms
- [[kl_divergence]] — relative entropy and its positivity
- [[mutual_information]] — the chain-rule consequence used for information gain
- [[information_gain]] — expected reduction in uncertainty
- [[epistemic_value]] — the term of expected free energy built on information gain
- [[coding_theory]] — operational meaning of entropy as compression limit

## References
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27, 379-423 and 623-656.
- Khinchin, A. I. (1957). *Mathematical Foundations of Information Theory*. Dover.
- Faddeev, D. K. (1956). On the concept of entropy of a finite probabilistic scheme. *Uspekhi Matematicheskikh Nauk*, 11(1), 227-231.
