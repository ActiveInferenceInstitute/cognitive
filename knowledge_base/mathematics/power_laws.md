---
title: Power Laws
description: Relationships in which one quantity varies as a power of another, their scale invariance and heavy tails, generating mechanisms, statistical pitfalls in fitting, and speculative links to criticality in the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - scale_invariance
  - heavy_tails
  - complex_systems
semantic_relations:
  - type: relates
    links:
      - "[[scale_invariance]]"
      - "[[phase_transitions]]"
      - "[[scale_free_networks]]"
      - "[[avalanche_dynamics]]"
      - "[[probability_distributions]]"
      - "[[free_energy_principle]]"
---

# Power Laws

## Overview
A power law is a relationship of the form $y=c\,x^{-\alpha}$. For a probability density, $p(x)\propto x^{-\alpha}$ for $x\ge x_{\min}$. On logarithmic axes, a power law appears as a straight line of slope $-\alpha$. The defining property is scale invariance: rescaling $x$ by a factor $\lambda$ multiplies $p$ by the constant $\lambda^{-\alpha}$, so there is no characteristic scale, in contrast to exponential or Gaussian laws.

## Heavy tails
For a continuous density with exponent $\alpha$, the mean is finite only if $\alpha>2$ and the variance only if $\alpha>3$. Distributions with exponents in the range $1<\alpha\le2$ have no finite mean, so sample averages do not stabilise. Rare, very large events therefore carry much of the total weight. Examples reported in the literature include city sizes, word frequencies, earthquake magnitudes, avalanche sizes in neural recordings and degree distributions of networks.

## Generating mechanisms
Several distinct mechanisms produce power laws: preferential attachment, in which growth is proportional to current size; multiplicative random processes with a lower boundary; critical phenomena at continuous phase transitions, where correlation length diverges; self-organised criticality in driven systems with threshold dynamics; and mixtures of exponentials across scales. A power law alone does not identify its mechanism.

## Statistical caution
Reporting a straight line on a log-log plot is not adequate evidence. Least-squares fits to binned histograms are biased. Clauset, Shalizi and Newman recommend maximum likelihood estimation of the exponent, estimation of $x_{\min}$, a goodness-of-fit test, and comparison against alternatives such as the lognormal and stretched exponential. Many claimed power laws fail these tests or cover only one or two decades.

## Relation to Active Inference
The connection is a research hypothesis and not a derived result. Proposals in the free energy literature link self-organising systems near non-equilibrium steady states to critical dynamics, and some neural data show avalanche statistics consistent with power laws, which supports models of brains operating near criticality. Whether free energy minimisation favours criticality, and whether observed neural power laws reflect it, is under investigation, and the evidence is mixed.

## Related concepts
- [[scale_invariance]] — the symmetry that characterises power laws
- [[phase_transitions]] — critical points are a classic source
- [[scale_free_networks]] — networks with power-law degree distributions
- [[avalanche_dynamics]] — cascades with power-law size distributions
- [[probability_distributions]] — the wider family within which heavy tails are classified

## References
- Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data. SIAM Review, 51, 661-703.
- Newman, M. E. J. (2005). Power laws, Pareto distributions and Zipf's law. Contemporary Physics, 46, 323-351.
- Sornette, D. (2006). Critical Phenomena in Natural Sciences (2nd ed.). Springer.
