---
title: Partition Function
description: "The partition function as the normalising constant of a Boltzmann distribution, with thermodynamic identities and its link to model evidence, softmax and free energy."
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - statistical_physics
  - normalization
  - free_energy
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/statistical_physics]]"
      - "[[knowledge_base/mathematics/softmax_function]]"
      - "[[knowledge_base/mathematics/temperature_parameter]]"
      - "[[knowledge_base/mathematics/variational_free_energy]]"
---

# Partition Function

## Overview
The partition function $Z$ is the normalising constant of a Boltzmann (Gibbs) distribution. It sums or integrates the unnormalised weights over all states, and its logarithm generates the thermodynamic quantities: mean energy, variance, entropy and free energy are all derivatives or combinations of $\ln Z$. The same object appears in probabilistic inference as the model evidence.

## Definition / Formalism
For states with energies $E_i$ and inverse temperature $\beta$,

$$p_i=\frac{e^{-\beta E_i}}{Z},\qquad Z=\sum_i e^{-\beta E_i}.$$

Standard identities (checked numerically on a three-level system):

$$\langle E\rangle=-\frac{\partial \ln Z}{\partial\beta},\qquad \mathrm{Var}(E)=\frac{\partial^2 \ln Z}{\partial\beta^2},\qquad F=-\frac{1}{\beta}\ln Z=\langle E\rangle-\frac{1}{\beta}S .$$

Variational characterisation: for any distribution $q$ over the states,

$$\sum_i q_i\,\beta E_i+\sum_i q_i\ln q_i=-\ln Z+D_{\mathrm{KL}}(q\,\|\,p)\;\ge\;-\ln Z,$$

with equality only at $q=p$. The Boltzmann distribution is therefore the minimiser of "energy minus entropy", and $-\ln Z$ is the minimum value.

The softmax is a Boltzmann distribution with $E_i=-x_i$ and $\beta=1/T$; its denominator is $Z$. Computing $\ln Z$ stably requires the log-sum-exp trick: for $x=(1000,1001)$ the shifted form $m+\ln\sum e^{x-m}$ gives 1001.313 where the naive $\exp$ overflows.

## Role in Active Inference
This is one of the strongest links in the statistical-physics family. Replace energy by $-\ln p(o,s)$ and the evidence $p(o)=\sum_s p(o,s)$ is a partition function of the joint. Variational free energy $F[q]=\mathbb{E}_q[\ln q(s)-\ln p(o,s)]$ then equals $-\ln p(o)+D_{\mathrm{KL}}(q\,\|\,p(s\mid o))$: with $A=\begin{pmatrix}0.9&0.1\\0.1&0.9\end{pmatrix}$, flat prior, $o=0$ and $q=(0.3,0.7)$, the identity gives $F\approx1.726$ against $-\ln p(o)=0.693$. Policy posteriors $\sigma(-\gamma G)$ are Boltzmann distributions over policies with precision $\gamma$ playing the role of $\beta$. The repo computes these normalisations directly for discrete spaces; no partition function is ever intractable there.

## Related concepts
- [[statistical_physics]] — where the partition function originates
- [[softmax_function]] — the normalised Boltzmann form used for policies and preferences
- [[temperature_parameter]] — $1/\beta$ in the exponent
- [[variational_free_energy]] — the bound on $-\ln Z$ above
- [[free_energy]] — thermodynamic and variational free energy compared
- [[numerical_stability]] — log-sum-exp in practice
