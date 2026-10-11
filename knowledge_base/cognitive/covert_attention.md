---
title: Covert Attention
description: The shifting of attentional focus to a location or feature without moving the eyes, how it is studied with spatial cueing, and its treatment as precision control in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - attention
  - spatial_attention
  - vision
semantic_relations:
  - type: relates
    links:
      - "[[spatial_attention]]"
      - "[[posner_attention]]"
      - "[[eye_movements]]"
      - "[[exogenous_attention]]"
      - "[[endogenous_attention]]"
---

# Covert Attention

## Overview
Covert attention is the selective processing of a part of the visual scene, or of another modality, without overt orienting of the eyes, head or body toward it. It contrasts with overt attention, in which the gaze is directed to the attended item. Hermann von Helmholtz described attending to a location in peripheral vision while holding the gaze fixed on a point, an early observation of the dissociation between attention and eye position. Covert shifts often precede an eye movement.

## Spatial cueing
The standard method is the cueing paradigm of Michael Posner (1980). Participants fixate a central point while a cue indicates a likely target location, and respond to a target presented at the cued location (valid trials) or elsewhere (invalid trials). Responses are faster and more accurate for valid than invalid trials, even though the eyes do not move, which shows that attention can be directed independently of gaze. Cues can be endogenous, such as a central arrow that is interpreted voluntarily, or exogenous, such as a peripheral flash that captures attention automatically (see [[endogenous_attention]] and [[exogenous_attention]]). Attention is reported to improve contrast sensitivity and spatial resolution at the attended location, and Carrasco reviewed the effects on perception.

## Neural basis and relation to eye movements
Covert attention engages a frontoparietal network that includes the frontal eye fields and the intraparietal cortex, together with the superior colliculus. These areas also control saccades, and stimulating or inactivating them affects covert attention. The premotor theory of attention proposes that covert shifts are saccade plans that are prepared but not executed. This claim is influential but debated, and findings suggest the two systems can be dissociated in some tasks (see [[frontal_eye_fields]]).

## Relation to Active Inference
The link is strong and has been modelled. Feldman and Friston proposed that attention is the optimisation of the precision of prediction errors, increasing the gain on sensory channels expected to be informative. They simulated the Posner paradigm to reproduce faster responses to validly cued targets and the effect of cue reliability. Covert attention is thus treated as an internal action that selects which evidence is to be weighted (see [[precision_weighting]]), while overt attention corresponds to actions that sample new data. Its neural implementation is still under test.

## Related concepts
- [[spatial_attention]] — attention directed to locations
- [[posner_attention]] — the cueing framework and attention networks
- [[eye_movements]] — overt orienting that covert shifts accompany
- [[exogenous_attention]] — stimulus-driven capture
- [[endogenous_attention]] — voluntary goal-driven orienting
- [[precision_weighting]] — the Active Inference mechanism for attentional gain

## References
- Posner, M. I. (1980). Orienting of attention. Quarterly Journal of Experimental Psychology, 32, 3-25.
- Carrasco, M. (2011). Visual attention: the past 25 years. Vision Research, 51, 1484-1525.
- Feldman, H., & Friston, K. J. (2010). Attention, uncertainty, and free-energy. Frontiers in Human Neuroscience, 4, 215.
