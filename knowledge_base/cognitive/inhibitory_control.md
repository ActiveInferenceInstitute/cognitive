---
title: Inhibitory Control
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - executive_function
  - control
  - action
aliases:
  - interference_control
  - inhibition
  - impulse_control
  - response_inhibition
semantic_relations:
  - type: relates
    links:
      - "[[executive_functions]]"
      - "[[cognitive_control]]"
      - "[[action_selection]]"
      - "[[active_inference]]"
---

# Inhibitory Control

## Overview
Inhibitory control is the ability to suppress prepotent responses, interference, or distraction in favour of goal-relevant behaviour. Miyake and colleagues identified response inhibition as one of three core [[executive_functions]], alongside updating of [[working_memory]] and shifting ([[cognitive_flexibility]]), although the unity and separability of these components is debated.

## Mechanism
Typical tasks measure different forms. In the stop-signal task, a go response is occasionally followed by a stop signal; Logan and Cowan's independent race model treats go and stop processes as racing, so the latency of the unobservable stopping process (stop-signal reaction time) can be estimated. Go/no-go tasks measure withholding; the Stroop and flanker tasks measure interference control, where an irrelevant dimension must be ignored.

Neural work implicates a frontal-basal ganglia network: the right inferior frontal cortex, the pre-supplementary motor area, and the subthalamic nucleus, through the so-called hyperdirect pathway, which can rapidly suppress motor output. Whether the right inferior frontal region is specific to inhibition or serves broader attentional detection is disputed. Inhibition interacts with [[conflict_monitoring]], which signals when more control is needed, and with [[attentional_control]]. Impaired response inhibition has been reported across several conditions and developmental stages, but these findings are group-level and not diagnostic.

## Role in Active Inference
There is no single canonical Active Inference model of inhibition. A natural reading is that stopping is a change in action selection: the agent infers that a policy previously favoured is no longer expected to minimise expected free energy, so the precision over that policy falls and an alternative (including "do nothing") takes over; see [[action_selection]] and [[policy_selection]]. Under this view inhibition is not a dedicated brake but an outcome of belief updating about which course of action is best, with precision ([[precision_weighting]]) determining how decisively prepotent habits are overridden. Prepotency itself corresponds to strong priors over habitual actions. This interpretation is a conceptual fit; detailed models have mostly been built in the reinforcement-learning and Bayesian-observer literature rather than in Active Inference specifically.

## Related concepts
- [[cognitive_control]] — the general capacity of which inhibition is part
- [[conflict_monitoring]] — detection of competing responses that calls for inhibition
- [[executive_functions]] — the family of control abilities
- [[task_switching]] — the shifting component
- [[self_regulation]] — inhibition applied to goal pursuit over longer timescales
- [[motor_control]] — the motor output being suppressed

## References
- Logan, G. D., & Cowan, W. B. (1984). On the ability to inhibit thought and action: a theory of an act of control. *Psychological Review*.
- Miyake, A., et al. (2000). The unity and diversity of executive functions and their contributions to complex "frontal lobe" tasks. *Cognitive Psychology*.
- Aron, A. R., Robbins, T. W., & Poldrack, R. A. (2014). Inhibition and the right inferior frontal cortex: one decade on. *Trends in Cognitive Sciences*.
