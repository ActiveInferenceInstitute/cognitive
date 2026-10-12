---
title: Sun Compass
description: The use of the sun's azimuth, compensated for time of day, to hold a heading, as studied in bees, desert ants, birds and other animals, and its place in navigation models.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - animal_navigation
  - insect_cognition
  - orientation
semantic_relations:
  - type: relates
    links:
      - "[[path_integration]]"
      - "[[social_insect_cognition]]"
      - "[[spatial_cognition]]"
      - "[[animal_cognition]]"
      - "[[bee_communication_systems]]"
---

# Sun Compass

## Overview
A sun compass is a mechanism by which an animal maintains a direction relative to the sun's azimuth. Because the sun moves across the sky, the mechanism needs compensation for time of day, supplied by an internal circadian clock. The sun compass has been demonstrated in many taxa, including honeybees, ants, birds, butterflies, beetles and some fishes and reptiles.

## Evidence
- **Bees.** Von Frisch (1967) showed that foragers communicate the direction of a food source in the waggle dance relative to gravity, which corresponds to the angle between the sun and the target, and that bees compensate for the sun's movement while dancing inside a dark hive (see [[bee_communication_systems]]).
- **Birds.** Kramer's experiments in the early 1950s on caged migratory starlings showed that they orient with the sun (Kramer, 1957). Later work by Hoffmann showed that starlings trained to a compass direction compensate for the sun's movement through an internal clock, and that shifting the clock with artificial light cycles shifted their chosen direction by a corresponding amount (Hoffmann, 1954).
- **Ants.** Desert ants such as *Cataglyphis* use the sun, and the pattern of polarised light in the sky when the sun itself is hidden, to measure direction while integrating path length for homing (Wehner, 2003; see [[path_integration]]).

## Mechanism
Insects detect the pattern of polarised skylight with specialised photoreceptors at the dorsal rim of the eye, and neural circuits in the central complex of the insect brain are thought to represent heading and combine compass cues with self-motion information. The relative weighting of the sun, polarisation, landmarks and the earth's magnetic field varies by species and conditions, and animals often combine several cues rather than relying on a single compass.

## Relation to Active Inference
The link is modest. Heading estimation from several noisy cues with learned reliabilities fits the cue-integration treatment of [[multisensory_integration]], in which cues are weighted by precision, and the compass is a standard example of state estimation. Models of insect navigation as active inference are an area of current research rather than an established result, and the neural implementation details above do not depend on the free-energy framework.

## Related concepts
- [[path_integration]] — dead reckoning that uses compass direction as input
- [[social_insect_cognition]] — navigation by foragers within the colony context
- [[spatial_cognition]] — representation of space and direction in animals
- [[cue_integration]] — combining several compass cues
- [[bee_communication_systems]] — the dance that encodes sun-relative direction

## References
- von Frisch, K. (1967). The Dance Language and Orientation of Bees. Harvard University Press.
- Kramer, G. (1957). Experiments on bird orientation and their interpretation. Ibis, 99, 196-227.
- Hoffmann, K. (1954). Versuche zu der im Richtungsfinden der Vogel enthaltenen Zeitschatzung. Zeitschrift fur Tierpsychologie, 11, 453-475.
- Wehner, R. (2003). Desert ant navigation: how miniature brains solve complex tasks. Journal of Comparative Physiology A, 189, 579-588.
