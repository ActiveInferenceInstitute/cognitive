---
title: Basal Ganglia
description: Interconnected subcortical nuclei that take part in action selection, habit formation and reinforcement learning, with an outline of their circuitry and of Active Inference proposals about policy selection.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - subcortical
  - action_selection
  - motor_control
semantic_relations:
  - type: relates
    links:
      - "[[action_selection]]"
      - "[[dopamine_system]]"
      - "[[motor_control]]"
      - "[[procedural_memory]]"
      - "[[reinforcement_learning]]"
wikidata: Q464210
---

# Basal Ganglia

## Overview
The basal ganglia are a group of interconnected subcortical nuclei at the base of the forebrain. The main components are the striatum (caudate nucleus, putamen and ventral striatum), the external and internal segments of the globus pallidus, the subthalamic nucleus, and the substantia nigra, which divides into pars compacta and pars reticulata. They receive input from most of the cerebral cortex and send output through the thalamus back to cortex and to brainstem targets. They are linked to action selection, habit formation and reward learning.

## Circuit organisation
Alexander, DeLong and Strick (1986) described several parallel, partly segregated loops running through the basal ganglia, usually labelled motor, oculomotor, associative and limbic according to the cortical area involved. Within a loop, striatal projection neurons act on the output nuclei (internal pallidum and substantia nigra pars reticulata) by two routes. In the standard simplified picture, the direct pathway, driven by neurons expressing D1 dopamine receptors, releases thalamus from tonic inhibition and so favours an action, while the indirect pathway, through the external pallidum and subthalamic nucleus and driven by D2-expressing neurons, increases inhibition. A hyperdirect route links cortex to the subthalamic nucleus. The selection hypothesis of Redgrave, Prescott and Gurney (1999) proposes that the output nuclei resolve competition between channels by disinhibiting only the winning action. The direct/indirect scheme is a simplification, as both pathways are active together during movement.

## Dopamine and disease
Dopamine from the substantia nigra pars compacta modulates striatal plasticity. Phasic dopamine activity resembles a reward prediction error (Schultz, Dayan and Montague, 1997). Loss of these neurons produces Parkinson's disease (see [[parkinsons]]), while degeneration of striatal neurons is the central feature of Huntington's disease. Altered striatal dopamine signalling is also discussed in [[addiction]].

## Relation to Active Inference
Active Inference proposals map policy selection onto basal ganglia circuitry. In the discrete-state process theory, posterior beliefs about policies are a softmax function of expected free energy, scaled by a precision term. Friston and colleagues (2014) suggested that striatal populations encode policy beliefs and that dopamine responses resemble changes in the precision of those beliefs, offering a reinterpretation of reward prediction error signals. This is an active research programme with simulated predictions, not an established anatomical result. Reinforcement-learning accounts of the same circuitry are the main competitor and make overlapping predictions.

## Related concepts
- [[action_selection]] — the computational problem the circuits are proposed to solve
- [[dopamine_system]] — the modulatory input that shapes striatal learning
- [[motor_control]] — the motor loop and its disorders
- [[procedural_memory]] — habit and skill storage involving the striatum
- [[parkinsons]] — disorder of nigrostriatal dopamine loss
- [[policy_selection]] — the Active Inference formulation of choosing among policies

## References
- Alexander, G. E., DeLong, M. R., & Strick, P. L. (1986). Parallel organization of functionally segregated circuits linking basal ganglia and cortex. Annual Review of Neuroscience, 9, 357-381.
- Redgrave, P., Prescott, T. J., & Gurney, K. (1999). The basal ganglia: a vertebrate solution to the selection problem? Neuroscience, 89, 1009-1023.
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. Science, 275, 1593-1599.
- Friston, K., et al. (2014). The anatomy of choice: dopamine and decision-making. Philosophical Transactions of the Royal Society B, 369, 20130481.
