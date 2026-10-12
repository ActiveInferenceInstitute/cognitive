---
title: Olfactory Processing
description: How odorants are detected by receptor neurons, organised into glomerular maps and read out in higher centres, with the insect version that underlies chemical communication in social insects.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - sensory_systems
  - olfaction
  - chemical_senses
semantic_relations:
  - type: relates
    links:
      - "[[sensory_processing]]"
      - "[[social_insect_cognition]]"
      - "[[chemical_communication]]"
      - "[[antennation]]"
      - "[[neural_coding]]"
---

# Olfactory Processing

## Overview
Olfactory processing is the neural handling of airborne chemical stimuli, from detection of odorant molecules to the perception and memory of odours. Vertebrates and insects use a similar architecture despite different evolutionary origins: a large family of receptor types, convergence of receptor neurons that express the same receptor onto discrete structures called glomeruli, and a second-stage network that relays the pattern to higher centres.

## Mechanism
In mammals, olfactory sensory neurons in the nasal epithelium each express one odorant receptor gene from a family of hundreds (Buck and Axel, 1991). Axons of neurons sharing a receptor converge on a small number of glomeruli in the olfactory bulb, so that an odour becomes a spatial pattern of glomerular activity. Mitral and tufted cells carry the signal to the piriform cortex and other targets. Unlike other senses, the main olfactory pathway does not pass through a thalamic relay before reaching cortex.

Insects detect odours with receptor neurons housed in sensilla on the antennae and other appendages. These project to the antennal lobe, which is organised into glomeruli and shaped by local inhibitory neurons, and then to the mushroom bodies and lateral horn. Odour identity is generally represented by combinations of receptor responses, so most odorants activate several receptor types and most receptors respond to several odorants.

## Social insects
In ants and other social insects, olfaction carries much of the communication that coordinates the colony. Cuticular hydrocarbons signal nestmate identity and caste, and pheromones signal alarm, trails and brood state (see [[chemical_communication]], [[alarm_pheromones]], [[trail_pheromones]]). Ants contact each other's antennae to sample these signals ([[antennation]]). Ant genomes are notable for large repertoires of odorant receptor genes, consistent with the weight of chemical signalling in their lives.

## Relation to Active Inference
The link is an application of general predictive-processing ideas, not a model specific to smell. Olfactory perception can be treated as inference about the hidden source of a noisy, turbulent chemical signal, and sniffing and orienting to a plume are actions that sample the environment to reduce uncertainty. Such models have been proposed in computational work on odour tracking, though the olfactory case is less developed than vision or audition.

## Related concepts
- [[sensory_processing]] — general principles of sensory transduction and coding
- [[neural_coding]] — combinatorial and population codes used for odour identity
- [[chemical_communication]] — signalling by chemical cues between organisms
- [[social_insect_cognition]] — cognition in colonies that rely on chemical cues
- [[antennation]] — contact sampling of chemical signals by ants

## References
- Buck, L., & Axel, R. (1991). A novel multigene family may encode odorant receptors: A molecular basis for odor recognition. Cell, 65, 175-187.
- Wilson, R. I., & Mainen, Z. F. (2006). Early events in olfactory processing. Annual Review of Neuroscience, 29, 163-201.
- Hölldobler, B., & Wilson, E. O. (1990). The Ants. Harvard University Press.
