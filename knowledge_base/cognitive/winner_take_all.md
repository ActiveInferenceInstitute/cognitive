---
title: Winner-Take-All
description: A competitive selection mechanism in which the strongest of several inputs or options is amplified while the rest are suppressed, covering its circuit form, decision models, and softmax relation to Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - selection
  - neural_networks
  - decision
semantic_relations:
  - type: relates
    links:
      - "[[decision_making]]"
      - "[[attractor_dynamics]]"
      - "[[neural_circuits]]"
      - "[[policy_selection]]"
      - "[[action_selection]]"
---

# Winner-Take-All

## Overview
Winner-take-all (WTA) is a form of competition among units in which the unit with the largest input becomes active and the others are silenced or strongly attenuated. The output is a selection, in the extreme case a single active unit. The principle appears in neural network theory, in models of perceptual and value-based choice, and in accounts of attention.

## Mechanism
The usual implementation is mutual inhibition. Each unit $i$ receives an input $I_i$ and inhibits its competitors, for example
$$\tau \dot{x}_i = -x_i + f\Big(I_i - w\sum_{j\neq i} x_j\Big).$$
When inhibition is strong enough, the dynamics have several stable states, one per input, and the state reached is determined by which input is largest, so the network behaves as an [[attractor_dynamics]] system with winner-selective fixed points. Grossberg analysed such recurrent competitive networks in the 1970s, and Maass (2000) showed that WTA circuits are computationally powerful building blocks. Variants include:
- **Hard WTA**, where one unit remains active.
- **Soft WTA**, where outputs are a graded, normalised function of the inputs, as with divisive normalisation or the softmax function.
- **k-WTA**, where the $k$ strongest units survive.
Cortical circuits with recurrent excitation and inhibition from interneurons are a plausible biological substrate (see [[neural_circuits]]).

## Use in decision models
Models of perceptual choice such as the leaky competing accumulator (Usher & McClelland, 2001) let evidence for each option accumulate under mutual inhibition until one crosses a threshold. This accounts for speed-accuracy trade-offs, and later extensions of the model address some context effects in choice (see [[decision_making]]). In attention, competition among representations resolved by a bias signal serves a similar role (see [[attention_mechanisms]]).

## Relation to Active Inference
In discrete-state Active Inference, policy posteriors are obtained by passing negative expected free energy, scaled by a precision parameter, through a softmax. The softmax is a soft winner-take-all: high precision sharpens the posterior towards a single policy, and low precision leaves it spread over several (see [[policy_selection]], [[precision_weighting]]). The connection is a formal analogy, since Active Inference does not require a neural WTA circuit, although recurrent competition is one candidate implementation of the message passing that realises it.

## Related concepts
- [[decision_making]] — competition as a model of choice
- [[attractor_dynamics]] — stable states that implement selection
- [[action_selection]] — mapping a policy posterior to a single action
- [[policy_selection]] — softmax over expected free energy
- [[attention_mechanisms]] — competition in attentional selection

## References
- Maass, W. (2000). On the computational power of winner-take-all. Neural Computation, 12, 2519-2535.
- Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: The leaky, competing accumulator model. Psychological Review, 108, 550-592.
