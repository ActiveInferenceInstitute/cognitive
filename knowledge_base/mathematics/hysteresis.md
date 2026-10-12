---
title: Hysteresis
description: The dependence of a system's state on its past history, producing different responses on increasing and decreasing input, with mechanisms, examples in physics, perception and collective behaviour.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - bistability
  - history_dependence
semantic_relations:
  - type: relates
    links:
      - "[[saddle_node_bifurcation]]"
      - "[[bifurcation_theory]]"
      - "[[basin_of_attraction]]"
      - "[[bistable_perception]]"
      - "[[swarm_intelligence]]"
wikidata: Q190837
---

# Hysteresis

## Overview
A system shows hysteresis when its output depends not only on the current input but also on the history of the input. Plotting output against a slowly varied input then gives a loop and not a single curve: the path followed when the input increases differs from the path followed when it decreases. The classic example is the magnetisation of a ferromagnet, which lags behind the applied field.

## Mechanism
A common origin is bistability. Suppose a system has a control parameter $\mu$ and, over a range of $\mu$, two stable states coexist, each with its own basin of attraction. As $\mu$ is raised, the system stays in the lower state until that state disappears at a fold, or saddle-node bifurcation, and then jumps to the upper state. When $\mu$ is lowered, the system remains in the upper state until it loses stability at a different value of $\mu$ and jumps back. The width of the loop is the distance between the two fold points. Slowly varying the parameter and tracking the state makes the history dependence visible, and the jumps are abrupt even though the parameter changes smoothly. Rate-independent hysteresis, as in magnetic and plastic materials, persists as the driving rate goes to zero, whereas loops caused by lags in a rapidly driven system shrink to a single curve.

## Where it appears
- **Perception**: ambiguous or bistable stimuli can be perceived in a way that depends on the preceding sequence, so the switching point differs between ascending and descending series.
- **Ecology**: a system that has crossed a tipping point may need a much larger reversal of conditions to recover, a pattern discussed in studies of regime shifts.
- **Collective behaviour**: group decisions with positive feedback, as in recruitment by social insects, can lock in to an early choice and resist change when conditions alter.

## Relation to Active Inference
Hysteresis is a general dynamical property, and Active Inference does not predict it directly. It is relevant where models of perception or choice have multiple stable beliefs, so that priors and recent history bias which interpretation is held. Such accounts are modelling proposals and need to be tested case by case.

## Related concepts
- [[saddle_node_bifurcation]] — the fold at which each branch of the loop ends
- [[bifurcation_theory]] — the study of parameter-dependent changes in stability
- [[basin_of_attraction]] — coexisting basins underlie history dependence
- [[critical_transitions]] — tipping points with delayed recovery
- [[bistable_perception]] — perceptual alternation and history effects
- [[swarm_intelligence]] — collective decisions that lock in

## References
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
- Scheffer, M., et al. (2001). Catastrophic shifts in ecosystems. Nature, 413, 591-596.
- Scheffer, M., et al. (2009). Early-warning signals for critical transitions. Nature, 461, 53-59.
