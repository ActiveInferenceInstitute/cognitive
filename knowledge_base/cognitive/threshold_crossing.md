---
title: Threshold Crossing
description: "Decisions triggered when accumulated evidence reaches a bound, with the drift-diffusion formulas for choice probability and decision time, the speed-accuracy trade-off, and Active Inference counterparts."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - decision-making
  - evidence-accumulation
  - dynamics
semantic_relations:
  - type: relates
    links:
      - "[[evidence_accumulation]]"
      - "[[decision_making]]"
      - "[[temporal_integration]]"
      - "[[action_selection]]"
---

# Threshold Crossing

## Overview
Threshold crossing is the mechanism in sequential-sampling models of decision-making by which a choice is committed: noisy evidence accumulates over time, and the first time it reaches a bound the corresponding response is triggered. The same idea describes spike generation at the level of a single neuron, where the membrane potential crosses a firing threshold. This page uses the decision sense.

## Definition and formalism
In the drift-diffusion model (Ratcliff), evidence $x$ evolves as $dx = v\,dt + \sigma\,dW$ from the start point $x=0$ until $|x| = a$. For drift $v$, noise $\sigma$, and symmetric bounds $\pm a$:
- the probability of terminating at the upper bound is $P = 1/(1 + e^{-2va/\sigma^2})$
- the mean decision time is $\langle T\rangle = \frac{a}{v}\tanh\!\left(\frac{av}{\sigma^2}\right)$

Both were checked against a simulated diffusion process and agree up to discretisation error. Raising $a$ increases accuracy and lengthens responses (the speed-accuracy trade-off), whereas raising $v$ (stronger or clearer evidence) improves both. Observed response times add a non-decision component for sensory and motor delays, and biases enter through start point or bound asymmetry. Variants include collapsing bounds, which reflect urgency, and race models with one accumulator per option.

Neural data support accumulation toward a bound: in monkeys, activity in lateral intraparietal area and related regions ramps up during motion discrimination, with the rate scaling with motion strength (Gold and Shadlen). Alternative accounts of ramping exist, so the mapping is not fully settled.

## Role in Active Inference
The threshold is not a primitive in Active Inference. Evidence accumulation corresponds to updating beliefs about hidden states as observations arrive, and commitment to a response corresponds to the policy posterior becoming sufficiently peaked ([[belief_updating]], [[policy_selection]]). Models therefore either add an explicit confidence criterion on the posterior or let the number of observation steps be fixed by the task. Because the posterior depends on precision, the effective bound can change with context, and the correspondence with a fixed-bound diffusion model holds only under particular assumptions.

## Related concepts
- [[evidence_accumulation]] - the process that precedes crossing
- [[decision_making]] - the setting for bound-based models
- [[value_computation]] - what the evidence may be about
- [[signal_detection]] - the static counterpart with a criterion
- [[temporal_integration]] - the integration underlying the drift

## References
- Ratcliff, R. (1978). A theory of memory retrieval. Psychological Review.
- Gold, J. I., & Shadlen, M. N. (2007). The neural basis of decision making. Annual Review of Neuroscience.
