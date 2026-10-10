---
title: Scene Perception
description: "How the visual system rapidly extracts the gist, layout, and category of a natural scene, the scene-selective cortical regions involved, and a predictive reading of scene construction."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - vision
  - perception
  - scenes
semantic_relations:
  - type: relates
    links:
      - "[[visual_perception]]"
      - "[[change_blindness]]"
      - "[[object_recognition]]"
      - "[[predictive_processing]]"
---

# Scene Perception

## Overview
Scene perception is the perception of whole visual environments (a kitchen, a beach, a street) as opposed to isolated objects. Observers can categorise a scene after very brief exposure, well before they have attended to individual objects. The topic overlaps with [[scene_understanding]]; here the emphasis is on the fast, largely feed-forward extraction of gist and spatial layout.

## Core findings
- **Gist.** Rapid serial visual presentation studies show that people can detect a described scene category at rates of several images per second. Global properties such as openness, naturalness, and mean depth support this, as in the spatial envelope description of scenes.
- **Layout before objects.** Spatial structure and statistical regularities of orientation and spatial frequency carry information about category without explicit object segmentation.
- **Scene-selective cortex.** Three regions are consistently reported: the parahippocampal place area (layout and category), the retrosplenial complex (orientation and navigation relative to a remembered environment), and the occipital place area (local geometry and navigational affordances).
- **Limited detail.** The impression of a rich, detailed percept coexists with poor memory for changes, as in [[change_blindness]]; the gist is retained while detail is sampled by [[visual_attention]].
- **Context effects.** Scene context speeds recognition of consistent objects and biases perception of ambiguous ones ([[object_recognition]]).

## Role in Active Inference
Gist works as a high-level prior: once the scene category is inferred, it constrains hypotheses about objects and locations at lower levels, and gaze is directed to locations that resolve the remaining uncertainty. Mirza and colleagues modelled this as scene construction combined with visual foraging, where saccades are chosen to maximise expected information gain ([[epistemic_foraging]]). This is a modelling programme with simulations rather than a settled account of scene-selective cortex.

## Related concepts
- [[scene_understanding]] - interpreting relations and events within the scene
- [[visual_perception]] - the broader field of which this is a part
- [[spatial_cognition]] - navigation and layout representations
- [[epistemic_foraging]] - active sampling of the scene
- [[peripheral_vision]] - the source of most gist information

## References
- Epstein, R., & Kanwisher, N. (1998). A cortical representation of the local visual environment. Nature.
- Oliva, A., & Torralba, A. (2001). Modeling the shape of the scene: a holistic representation of the spatial envelope. International Journal of Computer Vision.
- Mirza, M. B., Adams, R. A., Mathys, C. D., & Friston, K. J. (2016). Scene construction, visual foraging, and active inference. Frontiers in Computational Neuroscience.
