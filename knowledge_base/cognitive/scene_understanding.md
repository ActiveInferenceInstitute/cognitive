---
title: Scene Understanding
description: "Interpreting a visual scene as a structured set of objects, relations, and events, including context effects and how hierarchical generative models represent scene structure."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - vision
  - inference
  - context
semantic_relations:
  - type: relates
    links:
      - "[[scene_perception]]"
      - "[[hierarchical_inference]]"
      - "[[generative_model]]"
      - "[[prediction_error]]"
aliases:
  - scene_analysis
---

# Scene Understanding

## Overview
Scene understanding goes beyond recognising a scene's category ([[scene_perception]]). It is the construction of an interpretation: which objects are present, how they are arranged and related, what is happening, and what actions the scene affords. In computer vision it is posed as joint segmentation, detection, and relationship prediction; in cognitive science it is studied through context effects and event perception. The inbound links to this page mostly mean "context processing", which is the sense emphasised below.

## Core ideas
- **Structured interpretation.** A scene is parsed into entities and relations (support, containment, occlusion, agent-action-object). Violations of expected relations, such as an object floating without support, are detected quickly and slow recognition.
- **Context.** Object identity, size, and location are interpreted relative to scene context. Ambiguous shapes are resolved as the object the context makes likely, and consistent context facilitates detection.
- **Events and agents.** Understanding includes inferring causes and intentions of moving agents ([[biological_motion]], [[theory_of_mind]]).
- **Integration over fixations.** Because only a small region is seen sharply at a time, the interpretation is accumulated across eye movements and held in memory ([[change_blindness]] shows its limits).
- **Computational formulations.** Scene graphs and analysis-by-synthesis models represent a scene as latent objects with pose and relations, scored by how well a rendering explains the image.

## Role in Active Inference
Scene understanding is a natural application of hierarchical generative models: higher levels hold causes such as scene type and object configuration, lower levels hold features, and each level sends predictions downward and receives [[prediction_error]] upward ([[predictive_processing]]). Context effects then follow from priors at higher levels. The link is a modelling framework rather than a unique prediction; analysis-by-synthesis vision models share the same logic without invoking the free-energy principle.

## Related concepts
- [[hierarchical_inference]] - inference across levels of causes
- [[generative_model]] - the model class used to describe scene structure
- [[object_recognition]] - one component of the interpretation
- [[computational_vision]] - machine formulations of the problem
- [[predictive_processing]] - context as top-down prediction

## References
- Biederman, I., Mezzanotte, R. J., & Rabinowitz, J. C. (1982). Scene perception: detecting and judging objects undergoing relational violations. Cognitive Psychology.
