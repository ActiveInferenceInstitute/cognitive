---
title: Approach Avoidance
description: Motivational tendencies to move toward appealing stimuli or away from aversive ones, classic conflict models, brain systems involved, and how Active Inference treats such choices through preferences.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - motivation
  - emotion
  - decision_making
semantic_relations:
  - type: relates
    links:
      - "[[motivation]]"
      - "[[reward_processing]]"
      - "[[punishment]]"
      - "[[amygdala]]"
      - "[[decision_making]]"
---

# Approach Avoidance

## Overview
Approach and avoidance are two basic classes of motivated behaviour. Approach moves an organism toward a stimulus, typically one that is rewarding or promises a positive outcome. Avoidance moves it away from a stimulus that signals threat, punishment or loss. Many theories treat the two as separate systems instead of the ends of a single scale, so that a stimulus can evoke both tendencies at once and produce conflict.

## Models
- **Conflict models.** Lewin and later Miller described conflicts between two attractive options, two aversive options, or a single option with both attractive and aversive features. The approach-avoidance case produces hesitation, because the pull toward a goal and the push away from its cost can both increase as the goal comes closer, with the avoidance gradient steeper.
- **Reinforcement sensitivity.** Gray (1982) proposed a behavioural approach system sensitive to reward and a behavioural inhibition system sensitive to threat and conflict. Self-report measures derived from this idea are widely used (Carver and White, 1994), and later revisions changed the account of the inhibition system.
- **Hierarchical models.** Elliot (2006) distinguished approach and avoidance temperaments, goals and specific behaviours, and treated them as a hierarchy that connects personality to action.
- **Implicit tasks.** Reaction-time tasks measure faster arm flexion toward positive stimuli and extension from negative ones, though the effects depend on how stimuli are coded and are not always replicated.

## Neural systems
Dopaminergic circuits including the striatum are linked to approach and reward seeking (see [[basal_ganglia]] and [[dopamine_system]]). The [[amygdala]], prefrontal cortex and hippocampus contribute to threat evaluation and to resolving approach-avoidance conflict, and anxiety is often described as a disturbance of the balance (see [[anxiety]] and [[fear]]).

## Relation to Active Inference
In Active Inference the two tendencies are not separate modules. An agent holds prior preferences over outcomes, written as the C vector, and selects policies that minimise expected free energy, which combines expected preference satisfaction with information gain (see [[expected_free_energy]] and [[policy_selection]]). Moving toward a rewarding outcome and away from a feared one are both policies that bring observations closer to preferred ones. Approach-avoidance conflict has been modelled by giving one outcome both attractive and aversive features. This is a modelling framework rather than a demonstration that the brain computes this way.

## Related concepts
- [[motivation]] — the broader class of goal-directed drives
- [[reward_processing]] — neural handling of appetitive outcomes
- [[punishment]] — aversive consequences that drive avoidance
- [[fear]] — an emotion associated with avoidance
- [[decision_making]] — resolving conflict among options

## References
- Gray, J. A. (1982). The neuropsychology of anxiety: An enquiry into the functions of the septo-hippocampal system. Oxford University Press.
- Carver, C. S., & White, T. L. (1994). Behavioral inhibition, behavioral activation, and affective responses to impending reward and punishment: The BIS/BAS scales. Journal of Personality and Social Psychology, 67, 319-333.
- Elliot, A. J. (2006). The hierarchical model of approach-avoidance motivation. Motivation and Emotion, 30, 111-116.
