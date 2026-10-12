---
title: Spatial Cognition
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - cognitive
  - active_inference
aliases:
  - spatial_processing
  - spatial_intelligence
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/cognitive/enactive_cognition]]"
      - "[[knowledge_base/cognitive/spatial_attention]]"
      - "[[knowledge_base/cognitive/epistemic_foraging]]"
wikidata: Q2178611
---

# Spatial Cognition

## Overview

Spatial cognition covers how organisms represent where things are, where they themselves are, and how to get from one place to another. Core findings are well established. Tolman (1948) argued from rat behaviour for internal "cognitive maps". O'Keefe and Dostrovsky (1971) found hippocampal place cells that fire at particular locations, and Hafting and colleagues (2005) found grid cells in entorhinal cortex with periodic firing fields. Representations are described as allocentric when anchored to the world and egocentric when anchored to the body, and navigation requires converting between them. Interpreting this machinery as inference under a generative model is an active research programme, supported by simulations but not settled as an account of the neural data.

## Main Ideas

**Space as a generative model.** An agent needs hidden states for its position and heading, a transition model describing how movement changes them, and an observation model relating states to landmarks and sensory input. Localisation is then state inference, and mapping is learning the observation model. This is the same structure as a discrete POMDP, with locations as states and moves as actions ([[knowledge_base/mathematics/pomdp_framework]]).

**Path integration and correction.** Integrating self-motion gives a prior over position that drifts; landmark observations correct it. Grid-cell activity is often interpreted as supporting path integration. Casting this as prediction followed by error-driven correction is natural, but several other models of grid cells exist and the data do not single out the inference account.

**Exploration as epistemic foraging.** An agent in an unknown environment gains by choosing moves that reduce uncertainty about the layout, such as turning toward a poorly mapped region ([[knowledge_base/cognitive/epistemic_foraging]]). Under expected free energy, this information-seeking drive and the pragmatic drive toward a goal combine in one objective, so early behaviour is exploratory and later behaviour exploitative as uncertainty falls. Friston and colleagues (2012) analysed saccades as experiments chosen to resolve uncertainty about a visual scene, the same logic applied to gaze.

**Attention in space.** Selecting a region to attend to is a spatial choice with the same epistemic structure ([[knowledge_base/cognitive/spatial_attention]]).

**Enaction.** Enactive accounts emphasise that spatial knowledge is skilled engagement with an environment, not a stored map ([[knowledge_base/cognitive/enactive_cognition]]).

## Caveats

Discrete grid worlds scale poorly to realistic spaces because the state count grows with resolution and extent. Hierarchical and continuous representations are the usual remedy ([[knowledge_base/cognitive/hierarchical_models]]), and learning the map from experience, rather than specifying it, is the harder problem.

## In This Repository

Related pages: [[knowledge_base/cognitive/spatial_attention]], [[knowledge_base/cognitive/sensorimotor_coordination]], [[knowledge_base/systems/spatial_computing]], [[knowledge_base/systems/spatial_ecology]] and [[docs/guides/application/active_inference_spatial_applications]]. In code, `Simple_POMDP` and `Generic_POMDP` document discrete agents whose states can be read as locations, but neither contains a spatial map representation, place or grid-cell model, or navigation environment. The `Path_Network` entry is a simulation of networked agents adapting to changing water levels, not a navigation model.

## References

- Tolman, E. C. (1948). Cognitive maps in rats and men. *Psychological Review*.
- O'Keefe, J., Dostrovsky, J. (1971). The hippocampus as a spatial map: preliminary evidence from unit activity in the freely-moving rat. *Brain Research*.
- Hafting, T., Fyhn, M., Molden, S., Moser, M.-B., Moser, E. I. (2005). Microstructure of a spatial map in the entorhinal cortex. *Nature*.
- Friston, K., Adams, R. A., Perrinet, L., Breakspear, M. (2012). Perceptions as hypotheses: saccades as experiments. *Frontiers in Psychology*.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/cognitive/enactive_cognition]]
