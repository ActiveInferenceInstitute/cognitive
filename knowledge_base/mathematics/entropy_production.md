---
title: Entropy Production
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - stochastic_thermodynamics
  - markov_processes
  - irreversibility
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/non_equilibrium_steady_state]]"
      - "[[knowledge_base/mathematics/non_equilibrium_thermodynamics]]"
      - "[[knowledge_base/mathematics/kl_divergence]]"
      - "[[knowledge_base/mathematics/stochastic_processes]]"
wikidata: Q5380811
---

# Entropy Production

## Overview
Entropy production measures the irreversibility of a process: how distinguishable a trajectory is from its time reversal. For a stationary Markov process it is zero exactly when detailed balance holds and strictly positive when there are persistent probability currents, such as a net cycle through states. It is the quantity the second law constrains.

## Definition / Formalism
Take a discrete-time stationary Markov chain with transition probabilities $P_{ij}=P(i\mid j)$ and stationary distribution $\pi$. The entropy production per step (in units of $k_B$) is

$$\sigma=\sum_{i,j}\pi_j P_{ij}\ln\frac{\pi_j P_{ij}}{\pi_i P_{ji}} .$$

Equivalently $\sigma=D_{\mathrm{KL}}$ between the forward one-step joint $\pi_jP_{ij}$ and the reversed joint $\pi_iP_{ji}$, so $\sigma\ge0$, with equality iff $\pi_jP_{ij}=\pi_iP_{ji}$ (detailed balance). Numerically, for an asymmetric three-state chain the two expressions agreed to machine precision (0.03217), and for a symmetric chain the value was exactly zero.

For a continuous-time jump process with rates $w_{ij}$ (from $j$ to $i$), the Schnakenberg formula is

$$\dot S=\tfrac12\sum_{i\ne j}\big(w_{ij}\pi_j-w_{ji}\pi_i\big)\ln\frac{w_{ij}\pi_j}{w_{ji}\pi_i}\;\ge0 .$$

A three-state ring with forward rate $k_+$ and backward rate $k_-$ has uniform stationary distribution and $\dot S=(k_+-k_-)\ln(k_+/k_-)$; for $k_+=2,\,k_-=0.5$ this is $1.5\ln4\approx2.079$, which the general formula reproduces.

Fluctuation theorems refine the picture: $\sigma$ is the mean of a trajectory-level quantity that can be negative for individual trajectories but satisfies $\langle e^{-\Sigma}\rangle=1$.

## Role in Active Inference
Indirect and mostly theoretical. Treatments of the free-energy principle describe agents as systems at a non-equilibrium steady state, and a nonzero steady-state probability current is exactly what makes entropy production positive. That literature relates the flow of the dynamics to gradients of surprisal. The relation to entropy production is a physical interpretation layered on top of the inference scheme. The discrete belief-updating code in `code/tools/src` computes none of these quantities, and an agent's variational free energy should not be read as a thermodynamic dissipation rate without further assumptions.

## Related concepts
- [[non_equilibrium_steady_state]] — the regime where currents and entropy production persist
- [[non_equilibrium_thermodynamics]] — the physical framework
- [[fluctuation_theorems]] — exact constraints on the distribution of entropy production
- [[kl_divergence]] — entropy production as a forward/reverse divergence
- [[stochastic_processes]] — Markov chains and jump processes
- [[markov_properties]] — the memorylessness assumed above

## References
- Schnakenberg, J. (1976). Network theory of microscopic and macroscopic behavior of master equation systems. *Reviews of Modern Physics* 48, 571.
- Seifert, U. (2012). Stochastic thermodynamics, fluctuation theorems and molecular machines. *Reports on Progress in Physics* 75, 126001.
