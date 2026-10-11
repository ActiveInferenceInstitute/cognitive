---
title: Critical Transitions
description: Abrupt shifts of a complex system between alternative states when a threshold is crossed, the early-warning signals that may precede them, and the loose connections to Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - tipping_points
  - bifurcations
  - resilience
semantic_relations:
  - type: relates
    links:
      - "[[bifurcation_theory]]"
      - "[[hysteresis]]"
      - "[[phase_transitions]]"
      - "[[resilience_thinking]]"
      - "[[dynamical_systems]]"
---

# Critical Transitions

## Overview
A critical transition, often called a tipping point or regime shift, is a rapid change of a system from one state to a qualitatively different one after a slow change in a driver or a perturbation. Examples discussed in the literature include shifts of lakes from clear to turbid water, collapse of fisheries, the onset of epileptic seizures and sudden changes in financial markets. The central point is that gradual forcing can produce a disproportionately sudden response.

## Mechanism
In many models the transition is a bifurcation of the underlying dynamics. A system with a slowly varying control parameter $\mu$ rests near a stable equilibrium. At a fold (saddle-node) bifurcation that equilibrium disappears, and the state jumps to a distant attractor. Systems with alternative stable states show [[hysteresis]]: returning the driver to its earlier value does not reverse the shift, because recovery requires crossing a different threshold. Not every abrupt change is a bifurcation, since large noise or external shocks can also push a system across the boundary between basins.

## Early-warning signals
Near a fold bifurcation the dominant eigenvalue of the linearised dynamics approaches zero, so recovery from small perturbations slows. This critical slowing down is expected to appear as rising autocorrelation and rising variance in a time series of the system. Scheffer and colleagues reviewed these indicators and related measures such as skewness and flickering between states. The signals have been demonstrated in models and experiments, but their reliability in field data is limited by short records, noise, and transitions that do not follow the fold scenario.

## Relation to Active Inference
The connection is conceptual and the evidence is mixed. In the free energy principle, a system that persists occupies a set of attracting states, so loss of resilience and shifts between attractors can be described in its terms, and slowed dynamics near criticality fits discussions of brains operating near critical points. Some proposals apply this to transitions in mental states or neural dynamics. These are interpretive links; no result shows that early-warning statistics follow from Active Inference.

## Related concepts
- [[bifurcation_theory]] — the mathematics of qualitative change in dynamics
- [[hysteresis]] — path dependence of transitions between alternative states
- [[phase_transitions]] — the physical analogue involving order parameters
- [[resilience_thinking]] — frames the capacity to absorb shocks before shifting
- [[saddle_node_bifurcation]] — the fold that underlies many tipping points
- [[avalanche_dynamics]] — critical behaviour with scale-free event sizes

## References
- Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S. R., Dakos, V., Held, H., van Nes, E. H., Rietkerk, M., & Sugihara, G. (2009). Early-warning signals for critical transitions. Nature, 461, 53-59.
- Scheffer, M., Carpenter, S., Foley, J. A., Folke, C., & Walker, B. (2001). Catastrophic shifts in ecosystems. Nature, 413, 591-596.
