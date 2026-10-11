---
title: Avalanche Dynamics
description: Cascades of activity whose sizes follow power laws in systems poised near criticality, from sandpile models to neuronal avalanches in cortex, and how criticality is discussed in relation to free energy.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - self_organized_criticality
  - power_laws
  - neural_dynamics
semantic_relations:
  - type: relates
    links:
      - "[[power_laws]]"
      - "[[phase_transitions]]"
      - "[[cellular_automata]]"
      - "[[neural_dynamics]]"
      - "[[critical_transitions]]"
---

# Avalanche Dynamics

## Overview
An avalanche is a burst of activity in which a local event triggers further events, spreading through a system until it dies out. When the system sits near a critical point, avalanche sizes and durations have no typical scale and their distributions follow power laws, $P(s)\sim s^{-\tau}$. Avalanches have been studied in sandpiles, earthquakes, forest fires, magnetic materials and neural tissue, and are a standard signature in the study of critical phenomena.

## Self-organised criticality
Bak, Tang and Wiesenfeld introduced the sandpile model as an example of self-organised criticality. Grains are added one at a time to a lattice. A site that exceeds a threshold topples and passes grains to its neighbours, which may topple in turn. Without tuning any parameter, the system reaches a stationary state in which avalanches of all sizes occur, with power-law size statistics. The framework offered an explanation for scale-free fluctuations in nature, though its applicability to specific real systems, such as earthquakes, is debated.

## Branching and neuronal avalanches
A simple description is a branching process in which each active unit activates on average $\sigma$ others. When $\sigma<1$ activity dies quickly, when $\sigma>1$ it grows, and at $\sigma=1$ it is critical, with avalanche sizes distributed as a power law with exponent near $3/2$ in the mean-field case. Beggs and Plenz recorded spontaneous activity in cortical slice cultures and reported avalanches of local field potential events with power-law sizes, and the observation has been extended to other preparations. Interpretations of such data remain contested, since power laws can arise from other mechanisms and require careful statistical testing.

## Relation to Active Inference
The connection is speculative. Proposals in the free energy literature discuss why systems with a Markov blanket would be expected to hover near critical points, where sensitivity and information transmission are high, and criticality has been argued to benefit inference by balancing stability and responsiveness. These ideas are an active area of discussion and have not been established as consequences of the framework.

## Related concepts
- [[power_laws]] — the statistical form of avalanche size distributions
- [[phase_transitions]] — the critical point around which avalanches become scale-free
- [[cellular_automata]] — the lattice rule systems in which sandpiles are simulated
- [[neural_dynamics]] — the setting of neuronal avalanche recordings
- [[critical_transitions]] — abrupt shifts and critical slowing down
- [[complex_systems]] — the broader field for self-organised criticality

## References
- Bak, P., Tang, C., & Wiesenfeld, K. (1987). Self-organized criticality: an explanation of the 1/f noise. Physical Review Letters, 59, 381-384.
- Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. Journal of Neuroscience, 23, 11167-11177.
