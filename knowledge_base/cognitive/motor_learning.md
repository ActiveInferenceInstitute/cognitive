---
title: Motor Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - learning
  - motor
  - cerebellum
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[motor_control]]"
      - "[[procedural_memory]]"
---

# Motor Learning

## Overview
Motor learning is the set of processes by which practice produces relatively permanent improvements in movement: faster, more accurate, more consistent, or better adapted to new conditions. It is commonly divided into adaptation (recalibrating an existing movement to a perturbation), skill acquisition (building new movement patterns or sequences), and decision-level learning (choosing which action to take).

## Mechanism
Several partly separable mechanisms are described:

- **Error-based learning**: a forward model predicts the sensory consequence of a motor command, and the sensory prediction error drives a trial-by-trial correction. Classic paradigms are visuomotor rotation and force-field adaptation. A simple state-space model is $x_{n+1} = A x_n + B e_n$, where $x$ is the internal estimate, $e$ the error, $A$ a retention factor and $B$ a learning rate. The cerebellum is strongly implicated.
- **Reinforcement-based learning**: outcomes (success or failure) shape which movements are repeated, with basal ganglia circuits and dopaminergic signals implicated (see [[dopamine_system]], [[reward_processing]]).
- **Use-dependent learning**: repetition biases later movements toward those performed, independent of error.
- **Strategy and sequence learning**: explicit and implicit components that build chunks of movements (see [[sequence_learning]]).

Learning proceeds at multiple timescales, with a fast component that forgets quickly and a slow component that retains. Consolidation after practice, including during sleep, stabilizes the skill (see [[memory_consolidation]]).

## Role in Active Inference
Active inference casts movement as the fulfillment of proprioceptive predictions: descending signals are predictions about sensory input, and reflex arcs minimize the resulting prediction error, so no separate motor command stream is required (Adams, Shipp and Friston, 2013). Motor learning then becomes learning of the generative model that links intended outcomes to sensory consequences, including the mapping between proprioception, vision and action. This reproduces the idea of a forward model in the language of precision-weighted [[prediction_error]]. The framework offers an integrated account, but it is a theoretical reformulation: the well-supported empirical findings (cerebellar error learning, multiple timescales) predate it, and it does not yet uniquely predict the dissociations among motor learning types.

## Related concepts
- [[motor_control]] — the control problem that motor learning improves
- [[sensorimotor_coordination]] — coupling of perception and movement that is recalibrated
- [[procedural_memory]] — the memory system in which learned motor skills reside
- [[skill_acquisition]] — broader account of expertise development
- [[prediction_error]] — error signal driving adaptation
- [[reinforcement_learning]] — reward-driven component of motor learning

## References
- Wolpert, D. M., Ghahramani, Z., & Jordan, M. I. (1995). An internal model for sensorimotor integration. *Science*.
- Shadmehr, R., & Krakauer, J. W. (2008). A computational neuroanatomy for motor control. *Experimental Brain Research*.
- Adams, R. A., Shipp, S., & Friston, K. J. (2013). Predictions not commands: active inference in the motor system. *Brain Structure and Function*.
