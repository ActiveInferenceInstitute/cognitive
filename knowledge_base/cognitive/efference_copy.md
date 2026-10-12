---
title: Efference Copy
description: A copy of an outgoing motor command used to predict the sensory consequences of movement, and its role in forward models, sensory attenuation, agency and Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - sensorimotor
  - forward_models
  - motor_control
semantic_relations:
  - type: relates
    links:
      - "[[corollary_discharge]]"
      - "[[forward_models]]"
      - "[[motor_control]]"
      - "[[prediction_error]]"
      - "[[sensorimotor_coordination]]"
wikidata: Q1430868
---

# Efference Copy

## Overview
An efference copy is a copy of the efferent motor command that is retained in the nervous system and used to anticipate the sensory effects of the movement. The idea was formalised by von Holst and Mittelstaedt (1950) as the reafference principle. Sensory input that results from the animal's own movement is called reafference, while input from external changes is exafference. The copy allows the former to be predicted and discounted so that the latter stands out.

## Mechanism
In the standard account the command is passed to a forward model that outputs the expected sensory feedback. The prediction is compared with the actual signal, and the difference is used in several ways: to cancel or attenuate the expected component, to correct the ongoing movement, and to update the model when the mismatch persists. Wolpert, Ghahramani and Jordan (1995) provided experimental and computational evidence that human movement uses an internal forward model for state estimation, combining the predicted next state with delayed sensory feedback. The same comparison underlies adaptation to altered visual feedback and to forces applied to the arm.

## Agency and clinical relevance
Matches between predicted and received feedback are one cue for the sense of agency, the feeling of having caused an action. Frith and colleagues proposed that faulty prediction of the consequences of self-generated action could produce passivity experiences and misattributed inner speech in schizophrenia. That proposal has influenced later accounts, though it is a hypothesis with mixed empirical support.

## Terminology
The concept overlaps with [[corollary_discharge]]. Some authors use the terms synonymously, others reserve efference copy for the literal duplicate of the motor command.

## Relation to Active Inference
Active Inference replaces the command-and-copy architecture with descending proprioceptive predictions that reflex arcs fulfil, so there is no explicit motor command to copy. Adams, Shipp and Friston (2013) set out this "predictions, not commands" view for the motor system, and the same framework treats sensory attenuation as a reduction in the precision of incoming proprioceptive evidence during movement. The overlap with classical forward-model accounts is substantial, and the two are difficult to distinguish with behavioural data. The Active Inference version is an active theoretical programme rather than a settled replacement.

## Related concepts
- [[corollary_discharge]] — the broader class of internal signals to sensory areas
- [[forward_models]] — predictors of sensory consequences
- [[motor_control]] — the context in which the copy is generated
- [[prediction_error]] — the mismatch signal computed from the comparison
- [[sensorimotor_coordination]] — coupling between action and sensation

## References
- von Holst, E., & Mittelstaedt, H. (1950). Das Reafferenzprinzip. Naturwissenschaften, 37, 464-476.
- Wolpert, D. M., Ghahramani, Z., & Jordan, M. I. (1995). An internal model for sensorimotor integration. Science, 269, 1880-1882.
- Adams, R. A., Shipp, S., & Friston, K. J. (2013). Predictions not commands: active inference in the motor system. Brain Structure and Function, 218, 611-643.
