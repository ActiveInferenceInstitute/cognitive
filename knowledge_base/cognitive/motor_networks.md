---
title: Motor Networks
description: Motor networks are the interconnected cortical, subcortical, cerebellar, and spinal circuits that plan and execute movement, with an Active Inference reading of motor commands as proprioceptive predictions.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - motor
  - cortex
  - basal-ganglia
semantic_relations:
  - type: relates
    links:
      - "[[motor_control]]"
      - "[[motor_learning]]"
      - "[[sensorimotor_coordination]]"
---

# Motor Networks

## Overview
Movement is produced by a distributed set of interacting structures, not by a single motor centre. The motor networks comprise cortical areas that plan and command action, loops through the basal ganglia and thalamus that influence selection and initiation, the cerebellum that refines timing and coordination, and brainstem and spinal circuits that generate the final output. They also receive sensory and proprioceptive feedback, so motor processing is closed-loop.

## Components
- **Primary motor cortex (M1)**: layer 5 pyramidal neurons project through the corticospinal tract to spinal motor neurons, with a somatotopic organization. M1 activity relates to both movement kinematics and muscle forces, and the relative weighting is debated.
- **Premotor cortex and supplementary motor area**: involved in selecting and sequencing movements, with ventral premotor areas linked to object-directed grasping and the mirror-neuron system.
- **Posterior parietal cortex**: transforms sensory coordinates into motor plans for reaching and grasping ([[posterior_parietal]]).
- **Prefrontal cortex**: sets goals and rules that bias action selection ([[prefrontal]]).
- **Basal ganglia**: parallel cortico-striato-thalamo-cortical loops are implicated in selecting among actions and in reinforcement-dependent learning, with dopamine modulating them ([[dopamine_system]]). Parkinson's disease, associated with loss of dopaminergic input to the striatum, is a clinical illustration.
- **Cerebellum**: forward-model-like computations and error-based adaptation of movements ([[motor_learning]], [[forward_models]]).
- **Spinal cord and brainstem**: reflex arcs, central pattern generators for rhythmic movement, and descending modulation.

Networks reconfigure with learning: practice shifts activity from effortful, frontal-parietal engagement toward more automatic circuits ([[skill_acquisition]]).

## Role in Active Inference
Active Inference replaces the classical idea of motor commands with proprioceptive predictions. Descending signals specify expected sensory (proprioceptive) states, and spinal reflex arcs minimize the resulting prediction error by moving the muscles, so action is perception turned into a self-fulfilling prediction (Adams, Shipp, and Friston, 2013). Cortical hierarchies encode intentions at slow timescales and fine kinematics at fast ones, and attenuating the precision of sensory feedback is proposed to allow movement to be initiated against conflicting evidence. This is a process-theory proposal consistent with much of the anatomy, whereas competing optimal-control accounts explain the same data, and the critical empirical tests are still under way.

## Related concepts
- [[motor_control]] — computational and behavioural principles of movement
- [[motor_learning]] — adaptation and skill formation within these networks
- [[sensorimotor_coordination]] — integration of sensing and movement
- [[proprioception]] — the sensory modality that motor predictions target
- [[forward_models]] — predictive models attributed to cerebellum and cortex
- [[active_inference_for_control]] — control theory formulation of the framework

## References
- Adams, R. A., Shipp, S., & Friston, K. J. (2013). Predictions not commands: Active inference in the motor system. *Brain Structure and Function*, 218, 611-643.
