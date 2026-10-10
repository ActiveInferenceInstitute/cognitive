---
title: Information Dynamics
description: The study of how information is stored, transferred and modified in the time evolution of a system, using measures such as active information storage and transfer entropy, with Active Inference as a related but distinct framework.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - information_theory
  - transfer_entropy
  - complex_systems
semantic_relations:
  - type: relates
    links:
      - "[[information_flow]]"
      - "[[mutual_information]]"
      - "[[complex_systems]]"
      - "[[entropy_production]]"
---

# Information Dynamics

## Overview
Information dynamics treats the evolution of a dynamical or computational system as information processing, and asks where information is held, where it moves, and where it is combined. The best-known formal programme decomposes distributed computation into storage, transfer and modification (Lizier and colleagues). It applies to time series from neural recordings, cellular automata and agent-based models, and complements the geometric and thermodynamic views of dynamics.

## Core measures
For a process $X$ with past $X_t^{(k)}=(x_{t-k+1},\dots,x_t)$:
- **Entropy rate:** $h_\mu=\lim_{k\to\infty}H(x_{t+1}\mid X_t^{(k)})$, the irreducible uncertainty per step.
- **Active information storage:** $A_X=I\big(X_t^{(k)};x_{t+1}\big)$, how much of the next value is predictable from the process's own past.
- **Transfer entropy** (Schreiber, 2000): $T_{Y\to X}=I\big(y_t^{(l)};x_{t+1}\mid X_t^{(k)}\big)$, the extra predictability of $X$ gained from $Y$ beyond $X$'s own past. It is asymmetric and equals Granger causality for jointly Gaussian processes (up to a factor of two).
- **Local (pointwise) versions** assign each time step its own value, which can be negative (misinformative), and let one see where in space-time storage and transfer occur.

Cautions: estimates need long stationary data or bias-corrected estimators; transfer entropy measures predictive information flow and not causation, since common drivers and unobserved variables produce spurious transfer ([[causal_inference]]). Choice of embedding lengths $k$ and $l$ matters.

## Role in Active Inference
Active Inference is a different framework, in which dynamics minimise variational free energy, but the two interact. Information-theoretic quantities give a way to analyse the behaviour of fitted or simulated agents: how much their internal states store about the world, and whether information flows across a Markov blanket from external to internal states and back through action and sensation ([[information_flow]]). Entropy production and information flow are linked in stochastic thermodynamics, which is part of the physics underpinning the principle ([[entropy_production]]). Treating the principle itself as a statement about information storage or transfer would exceed what its derivation supports.

## Related concepts
- [[information_flow]] — transfer-entropy and directed information between variables
- [[mutual_information]] — building block for storage and transfer measures
- [[entropy]] — uncertainty measure underlying entropy rates
- [[complex_systems]] — domain where distributed computation is studied
- [[information_theory]] — the mathematical base
- [[markov_blanket]] — structure across which information flow can be analysed

## References
- Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85, 461-464.
- Lizier, J. T., Prokopenko, M., & Zomaya, A. Y. (2014). A framework for the local information dynamics of distributed computation in complex systems. In *Guided Self-Organization: Inception* (pp. 115-158). Springer.
