---
title: Primary Motor Cortex
description: The precentral cortical area that gives rise to a major share of corticospinal output, its body map, its encoding of movement variables and its treatment as a source of predictions in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - motor_system
  - cortex
  - movement
semantic_relations:
  - type: relates
    links:
      - "[[motor_cortex]]"
      - "[[premotor_cortex]]"
      - "[[motor_control]]"
      - "[[motor_units]]"
      - "[[population_vector]]"
      - "[[proprioception]]"
---

# Primary Motor Cortex

## Overview
The primary motor cortex, abbreviated M1, occupies the precentral gyrus of the frontal lobe and corresponds to Brodmann area 4. It is the cortical area from which relatively weak electrical stimulation most readily evokes movement, and it is a main source of the descending corticospinal tract. In primates, some corticospinal neurons from M1 make direct connections with spinal motor neurons, which is thought to be important for fine control of the fingers. It is one component of the broader motor cortex ([[motor_cortex]]), together with [[premotor_cortex]] and the supplementary motor areas.

## Somatotopy
Electrical stimulation studies in humans undergoing neurosurgery showed an orderly map of the body across the precentral gyrus, with the foot and leg medially, then the trunk, arm and hand, and the face laterally (Penfield and Boldrey, 1937). The map is often drawn as a motor homunculus. Representation is proportional to the precision of control and not to body size, so the hand and face occupy large areas. Later work showed that the map is not a strict one-to-one layout: representations overlap, and they are modifiable by use, injury and training.

## Encoding of movement
Neurons in M1 carry information about several movement parameters, including direction, force and velocity. Georgopoulos and colleagues showed that individual cells are broadly tuned to the direction of arm movement, and that a weighted sum across cells predicts the direction ([[population_vector]]). Subsequent work suggests that population activity during reaching is better described as a dynamical system, with rotational patterns that generate the muscle commands (Churchland et al., 2012). Which description is better is still discussed.

## Relation to Active Inference
The connection is a worked proposal. In the Active Inference account, M1 is an agranular cortex in which prediction rather than prediction error dominates, and it sends descending signals that specify expected proprioceptive input. Spinal reflex arcs then produce movement to fulfil these predictions (Shipp, Adams and Friston, 2013; Adams, Shipp and Friston, 2013). This reverses the usual picture in which cortex issues motor commands. The proposal is consistent with the sparse granular layer of agranular motor cortex, but other control theories, including optimal feedback control, explain much of the behaviour, and the question of which better describes M1 remains open (see [[motor_control]] and [[proprioception]]).

## Related concepts
- [[motor_cortex]] — the broader set of cortical motor areas
- [[premotor_cortex]] — the neighbouring area that prepares and selects movements
- [[motor_units]] — the spinal motor neurons and muscle fibres that M1 ultimately drives
- [[population_vector]] — a method for decoding movement direction from M1 activity
- [[proprioception]] — sensory feedback about limb position that M1 predicts

## References
- Penfield, W., & Boldrey, E. (1937). Somatic motor and sensory representation in the cerebral cortex of man as studied by electrical stimulation. Brain, 60, 389-443.
- Churchland, M. M., Cunningham, J. P., Kaufman, M. T., Foster, J. D., Nuyujukian, P., Ryu, S. I., & Shenoy, K. V. (2012). Neural population dynamics during reaching. Nature, 487, 51-56.
- Shipp, S., Adams, R. A., & Friston, K. J. (2013). Reflections on agranular architecture: Predictive coding in the motor cortex. Trends in Neurosciences, 36, 706-716.
