---
title: Prefrontal Cortex
description: The prefrontal cortex supports goal-directed control, working memory, planning, and valuation through interacting subregions, and is the main candidate substrate for deep temporal models in Active Inference.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - cortex
  - executive-function
  - control
semantic_relations:
  - type: relates
    links:
      - "[[dorsolateral_prefrontal]]"
      - "[[medial_prefrontal]]"
      - "[[executive_functions]]"
---

# Prefrontal Cortex

## Overview
The prefrontal cortex (PFC) is the anterior part of the frontal lobe, anterior to premotor cortex. Its defining feature is dense, reciprocal connectivity with nearly all sensory, motor, limbic, and subcortical systems, which positions it to coordinate behaviour with internal goals. It is disproportionately expanded in primates and is among the latest cortical regions to mature, with development continuing through adolescence.

## Subregions and functions
- **Dorsolateral PFC**: working memory, rule maintenance, and planning ([[dorsolateral_prefrontal]], [[working_memory]]).
- **Ventrolateral PFC**: selection and retrieval of information, and response inhibition ([[inhibitory_control]]).
- **Medial PFC and anterior cingulate**: self-referential thought, social cognition, performance monitoring, and effort-cost evaluation ([[medial_prefrontal]], [[anterior_cingulate]]).
- **Orbitofrontal and ventromedial PFC**: value, outcome expectation, and emotion regulation ([[orbitofrontal]]).
- **Frontopolar PFC**: integration of sub-goals and branching tasks.

**Integrative theories.** Miller and Cohen (2001) proposed that PFC maintains representations of goals and rules that bias processing in other regions toward task-relevant pathways. Koechlin and colleagues (2003) reported evidence for a hierarchical organization along the rostro-caudal axis, with more anterior areas handling more abstract control signals and longer timescales. Lesions produce impairments in planning, flexibility, and social conduct, while basic perception and memory are comparatively preserved. A concern is that "executive" labels are loose, and the unitary central executive notion is rejected by most current accounts ([[executive_functions]], [[cognitive_control]]).

## Role in Active Inference
Hierarchical generative models with deeper temporal structure are a natural description of the PFC functions listed above. Higher levels encode slowly changing goals and contexts that set priors for lower levels, and the anterior-to-posterior gradient of abstraction parallels the model hierarchy ([[hierarchical_inference]]). Working-memory maintenance can be modelled as sustained beliefs over hidden states, planning as inference over policies, and cognitive control as precision allocation across competing hypotheses ([[precision_weighting]]). Dopaminergic modulation of PFC is associated with the precision of policy beliefs ([[dopamine_system]]). These are model-based interpretations, and the specific cortical implementation of policy inference remains a hypothesis under study.

## Related concepts
- [[dorsolateral_prefrontal]] — subregion for working memory and rules
- [[medial_prefrontal]] — subregion for self and social processing
- [[orbitofrontal]] — subregion for valuation
- [[executive_functions]] — functions attributed to PFC
- [[cognitive_control]] — regulation of thought and action by goals
- [[working_memory]] — maintenance function most strongly linked to PFC
- [[hierarchical_inference]] — computational scheme linked to PFC organization

## References
- Miller, E. K., & Cohen, J. D. (2001). An integrative theory of prefrontal cortex function. *Annual Review of Neuroscience*, 24, 167-202.
- Koechlin, E., Ody, C., & Kouneiher, F. (2003). The architecture of cognitive control in the human prefrontal cortex. *Science*, 302, 1181-1185.
