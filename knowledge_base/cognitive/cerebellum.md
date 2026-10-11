---
title: Cerebellum
description: The hindbrain structure containing most of the brain's neurons, central to motor coordination and error-driven learning, with growing evidence for roles in cognition and the internal-model view of its function.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neuroanatomy
  - motor_control
  - internal_models
semantic_relations:
  - type: relates
    links:
      - "[[motor_control]]"
      - "[[motor_learning]]"
      - "[[forward_models]]"
      - "[[skill_acquisition]]"
      - "[[sensorimotor_coordination]]"
aliases:
  - cerebellar
---

# Cerebellum

## Overview
The cerebellum is a highly folded structure at the back of the brain, attached to the brainstem. Although it occupies only a small fraction of brain volume, it contains the large majority of the brain's neurons, mostly small granule cells. Its classical role is the coordination and calibration of movement: patients with cerebellar damage retain the ability to move but show ataxia, intention tremor, dysmetria (misjudged distance) and impaired adaptation to altered conditions.

## Circuit and learning
Input arrives through mossy fibres, which excite granule cells whose parallel fibres contact the dendrites of Purkinje cells, the sole output of the cerebellar cortex. A second input, the climbing fibre from the inferior olive, contacts each Purkinje cell and was proposed to carry error or teaching signals. Marr (1969) proposed that climbing-fibre activity modifies the strength of parallel fibre synapses, and later work by Albus and Ito developed this into the theory that the change is a long-term depression, so that the cerebellum learns associations between context and corrective output. This supports fast, accurate movements, and eye-blink conditioning and adaptation of the vestibulo-ocular reflex are standard experimental models (see [[motor_learning]]).

## Internal models and cognition
A prominent view holds that the cerebellum implements internal models. A forward model predicts the sensory consequences of a motor command, allowing movement to be corrected before feedback arrives and allowing self-generated sensations to be distinguished from external ones (Wolpert et al., 1998; Ito, 2008; see [[forward_models]]). Anatomical loops with prefrontal and parietal cortex have led to proposals that the same machinery supports prediction in thought and language, and cerebellar lesions can produce deficits in executive function, language and affect. The strength of the evidence for specific non-motor mechanisms is still debated.

## Relation to Active Inference
The forward-model account is the closest link. In Active Inference, movement results from descending proprioceptive predictions that reflex arcs fulfil, and prediction errors are minimised across a hierarchy that includes cerebellar circuits (see [[prediction_error]]). Sensory attenuation of self-generated input is explained by lowering the precision of predicted sensations during action. Casting cerebellar learning as precision-weighted error minimisation is a modelling proposal consistent with the anatomy, not a settled description of the circuit, and optimal-control accounts compete with it.

## Related concepts
- [[motor_control]] — the control problem the cerebellum helps solve
- [[motor_learning]] — error-driven adaptation of movement
- [[forward_models]] — predictive models attributed to cerebellar circuits
- [[skill_acquisition]] — gradual refinement of practised actions
- [[sensorimotor_coordination]] — coupling of sensation and movement

## References
- Marr, D. (1969). A theory of cerebellar cortex. Journal of Physiology, 202, 437-470.
- Wolpert, D. M., Miall, R. C., & Kawato, M. (1998). Internal models in the cerebellum. Trends in Cognitive Sciences, 2, 338-347.
- Ito, M. (2008). Control of mental activities by internal models in the cerebellum. Nature Reviews Neuroscience, 9, 304-313.
