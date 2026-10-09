---
title: Conflict Monitoring
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - cognitive_control
  - executive_function
  - performance_monitoring
semantic_relations:
  - type: relates
    links:
      - "[[cognitive_control]]"
      - "[[anterior_cingulate]]"
      - "[[inhibitory_control]]"
      - "[[active_inference]]"
---

# Conflict Monitoring

## Overview
Conflict monitoring is a theory, and the process it describes, in which the brain detects competition between incompatible response tendencies and uses that signal to increase [[cognitive_control]]. It was formalised by Botvinick, Braver, Barch, Carter, and Cohen in 2001 as a solution to the question of how a control system knows when to engage.

## Mechanism
In tasks such as Stroop (naming the ink colour of a colour word), flanker, and Simon, incongruent trials activate two competing responses, one driven by the automatic stimulus dimension and one by the task goal, and responses are slower and more error-prone. The model proposes that a monitoring module detects such coactivation, quantified as the Hopfield energy: the sum over pairs of response units of the product of their activations and the (inhibitory) weight between them. This conflict signal is passed to a control module, which increases top-down attention to the task-relevant dimension on subsequent trials.

The model explains conflict adaptation, the Gratton effect: interference is smaller on trials following an incongruent trial than following a congruent one. Neural evidence implicates the [[anterior_cingulate]] as the monitor, with greater activity on high-conflict trials, and lateral prefrontal cortex ([[dorsolateral_prefrontal]]) as the controller. The theory has been challenged. Brown and Braver argued that the ACC learns the likelihood of error rather than conflict as such, and the expected value of control account reframes the ACC as computing the payoff of control. Some adaptation effects can also be produced by low-level stimulus-repetition confounds, which require careful trial sequencing to rule out. Conflict is also signalled by the error-related negativity in EEG, which is generated in the same medial frontal region.

## Role in Active Inference
Conflict can be redescribed as uncertainty about which policy to follow, or high entropy over competing action plans, in which no policy is clearly favoured by expected free energy ([[policy_selection]], [[action_selection]]). A system that tracks this uncertainty could increase the precision of goal-relevant beliefs or allocate more computation, which parallels the conflict-driven increase in control ([[precision_weighting]], [[attentional_control]]). The Hopfield-energy measure and the entropy-over-policies measure are not equivalent, and I am not aware of a direct, quantitative derivation of conflict monitoring within Active Inference; the relation is a conceptual correspondence that points to a modelling opportunity rather than an established result.

## Related concepts
- [[anterior_cingulate]] — the region most often proposed as the conflict monitor
- [[cognitive_control]] — the control adjustments that conflict triggers
- [[inhibitory_control]] — suppression of the prepotent response
- [[attentional_control]] — top-down biasing toward relevant stimuli
- [[task_switching]] — another paradigm where conflict and adaptation are studied
- [[prediction_error]] — a related error signal used in alternative accounts

## References
- Botvinick, M. M., Braver, T. S., Barch, D. M., Carter, C. S., & Cohen, J. D. (2001). Conflict monitoring and cognitive control. *Psychological Review*.
- Gratton, G., Coles, M. G. H., & Donchin, E. (1992). Optimizing the use of information: strategic control of activation of responses. *Journal of Experimental Psychology: General*.
- Brown, J. W., & Braver, T. S. (2005). Learned predictions of error likelihood in the anterior cingulate cortex. *Science*.
