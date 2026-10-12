---
title: Spotlight Mechanism
description: The proposal that visual attention operates like a beam that enhances processing within a selected region of space, with the zoom-lens variant, supporting evidence, criticisms and precision-based accounts.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - attention
  - visual_attention
  - selection
semantic_relations:
  - type: relates
    links:
      - "[[spotlight]]"
      - "[[posner_attention]]"
      - "[[covert_attention]]"
      - "[[spatial_attention]]"
      - "[[object_based_attention]]"
      - "[[precision_weighting]]"
---

# Spotlight Mechanism

## Overview
The spotlight mechanism is a metaphor and a family of models in which visual attention selects a limited region of the visual field for enhanced processing, while stimuli outside the region receive reduced processing. The idea was developed from experiments in which a cue indicating a likely target location speeded responses and improved accuracy for targets that appeared there, even though the eyes remained fixed. This dissociation of attention from eye position is the basis of the term covert attention.

## Evidence and extensions
Posner (1980) used cues presented before a target and found faster detection at cued than at uncued locations, with costs at invalid locations. The result is typically interpreted as attention moving to the cued position, either voluntarily in response to a central cue or reflexively in response to a peripheral one. Eriksen and St. James (1986) proposed a zoom-lens variant, in which the attended region can be widened or narrowed with a trade-off: a larger region is processed with lower resolution or efficiency. Neurophysiological work found that responses of visual neurons whose receptive fields overlap the attended location are enhanced, consistent with a gain or competition account (Desimone and Duncan, 1995).

The spotlight is not the only account. Evidence for object-based selection shows that attention can spread across an object, including regions beyond a simple contiguous zone, and that it can be allocated to features such as colour across the whole field (see [[object_based_attention]], [[feature_based_attention]]). Attention can also be split across non-contiguous locations under some conditions. For these reasons, the spotlight is generally treated as a useful description of spatial selection rather than a literal mechanism.

## Relation to Active Inference
Active Inference and predictive processing recast spatial attention as the optimisation of precision. Feldman and Friston (2010) proposed that attention corresponds to increasing the precision, or inverse variance, assigned to prediction errors from a selected channel, which has the effect of a gain increase for the attended region. In this reading the spotlight is an emergent effect of expected precision, the same inference that explains cueing effects in a Posner task. This account has been modelled and applied to cueing paradigms; evidence that distinguishes it from other gain-control models is still being gathered (see [[precision_weighting]]).

## Related concepts
- [[spotlight]] — the broader metaphor and its history
- [[posner_attention]] — cueing paradigm and the attentional networks it motivated
- [[covert_attention]] — attention without eye movement
- [[spatial_attention]] — selection by location
- [[object_based_attention]] — a competing account of the unit of selection
- [[precision_weighting]] — the Active Inference mechanism for attentional gain

## References
- Posner, M. I. (1980). Orienting of attention. Quarterly Journal of Experimental Psychology, 32, 3-25.
- Eriksen, C. W., & St. James, J. D. (1986). Visual attention within and around the field of focal attention: A zoom lens model. Perception & Psychophysics, 40, 225-240.
- Desimone, R., & Duncan, J. (1995). Neural mechanisms of selective visual attention. Annual Review of Neuroscience, 18, 193-222.
- Feldman, H., & Friston, K. J. (2010). Attention, uncertainty, and free-energy. Frontiers in Human Neuroscience, 4, 215.
