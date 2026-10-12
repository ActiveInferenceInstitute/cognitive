---
title: Priority Maps
description: Topographic maps that combine stimulus salience with behavioural relevance to specify which location should be attended or fixated next, and their relation to expected information gain in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - attention
  - visual_attention
  - saliency
semantic_relations:
  - type: relates
    links:
      - "[[salience]]"
      - "[[attention_mechanisms]]"
      - "[[spatial_attention]]"
      - "[[overt_attention]]"
      - "[[frontal_eye_fields]]"
      - "[[visual_search]]"
---

# Priority Maps

## Overview
A priority map is a spatially organised representation in which each location is assigned a value that reflects how strongly it should attract attention or gaze. The value combines how much a location stands out from its surroundings, through bottom-up salience, with how relevant it is to current goals, through top-down guidance, and may also include reward history and recent experience. A winner-take-all or competitive process on the map then selects the next target. The term was developed to describe the mixture of signals found in oculomotor and attention-related brain areas (Fecteau and Munoz, 2006).

## Saliency maps and priority maps
Koch and Ullman (1985) proposed a saliency map, a single map that codes conspicuity by combining feature contrasts across dimensions such as colour, orientation and luminance, together with a winner-take-all stage that selects the most conspicuous location. The priority map extends the idea. Because behaviour is often driven by relevance and not just conspicuity, a purely stimulus-driven map cannot explain why observers find a target defined by a known feature, or ignore a conspicuous distractor. Priority therefore integrates both sources.

## Neural candidates
Neurons with the properties of a priority map have been recorded in several areas, including the superior colliculus, the frontal eye fields ([[frontal_eye_fields]]), the lateral intraparietal area and parts of the pulvinar. Serences and Yantis (2006) discussed how a priority signal in frontoparietal cortex could bias activity in visual areas to enhance the processing of selected locations. Which area, if any, holds the single priority map is disputed, and some accounts propose distributed maps that interact.

## Relation to Active Inference
The link is a developed modelling correspondence. In Active Inference, the choice of where to look is guided by expected free energy, in which epistemic value measures the expected reduction of uncertainty about the scene and pragmatic value measures the agreement of expected outcomes with preferences. Parr and Friston (2019) argued that salience can be defined in this framework as the expected information gain of sampling a location, while relevance enters through prior preferences, so that a map of expected free energy over locations plays the role of a priority map. This offers a formal reading of the construct, though it is one of several, and its predictions for neural recordings have not been tested in detail.

## Related concepts
- [[salience]] — the stimulus-driven component of priority
- [[attention_mechanisms]] — the general set of processes that select information
- [[spatial_attention]] — the selection of locations that priority maps guide
- [[overt_attention]] — the selection of gaze targets from the map
- [[visual_search]] — a task in which priority maps are inferred from performance

## References
- Koch, C., & Ullman, S. (1985). Shifting selective attention: Towards the underlying neural circuitry. Human Neurobiology, 4, 219-227.
- Itti, L., & Koch, C. (2001). Computational modelling of visual attention. Nature Reviews Neuroscience, 2, 194-203.
- Fecteau, J. H., & Munoz, D. P. (2006). Salience, relevance, and firing: A priority map for target selection. Trends in Cognitive Sciences, 10, 382-390.
- Parr, T., & Friston, K. J. (2019). Attention or salience? Current Opinion in Psychology, 29, 1-5.
- Serences, J. T., & Yantis, S. (2006). Selective visual attention and perceptual coherence. Trends in Cognitive Sciences, 10, 38-45.
