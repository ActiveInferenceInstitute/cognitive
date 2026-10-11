---
title: Posner Attention
description: Michael Posner's spatial cueing paradigm and network model of attention, covering orienting, alerting and executive control, and their reinterpretation as precision control in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - attention
  - orienting
  - cueing
semantic_relations:
  - type: relates
    links:
      - "[[attention_mechanisms]]"
      - "[[spatial_attention]]"
      - "[[exogenous_attention]]"
      - "[[endogenous_attention]]"
      - "[[precision_weighting]]"
---

# Posner Attention

## Overview
Posner attention refers to the experimental paradigm and theoretical framework developed by Michael Posner and colleagues for studying how attention is directed in space and how it is organised in the brain. The best known element is the spatial cueing task. The framework also includes a division of attention into three functional networks.

## Spatial cueing
In the cueing task (Posner, 1980) a participant fixates a central point and responds to a target that appears in one of two peripheral locations. A cue shortly before the target indicates a location. On valid trials the target appears where cued, and on invalid trials at the other location. Responses are faster and more accurate on valid than on invalid trials, and the difference, the validity effect, is taken as a measure of orienting.
- **Endogenous cues** are central symbols, such as arrows, that must be interpreted. They produce voluntary orienting that builds over several hundred milliseconds (see [[endogenous_attention]]).
- **Exogenous cues** are peripheral flashes. They capture attention automatically and quickly (see [[exogenous_attention]]). At longer delays the effect can reverse, a phenomenon called inhibition of return.
Orienting can occur covertly, without eye movements, which is why the task is used to study attention separately from gaze.

## Network model
Posner and Petersen (1990) proposed that attention comprises separable systems for alerting, orienting and executive control. Alerting maintains readiness and is associated with the noradrenergic system and right-hemisphere regions. Orienting selects information from sensory input and involves parietal and frontal areas, including the frontal eye fields (see [[frontal_eye_fields]], [[posterior_parietal]]). Executive control resolves conflict among responses and involves the anterior cingulate and lateral prefrontal cortex. The network distinction is influential, although the anatomy and independence of the networks have been refined by later imaging work.

## Relation to Active Inference
Feldman and Friston (2010) proposed that attention is the optimisation of precision, the expected reliability of sensory evidence in a hierarchical generative model, and used a Posner cueing task to simulate the validity effect. A valid cue raises the precision assigned to the cued location, so prediction errors from that location have greater influence, giving faster belief updating; invalid cues require a costly reallocation. This is a formal model reproducing the core behavioural effect, with neurobiological suggestions that remain to be tested (see [[precision_weighting]]).

## Related concepts
- [[attention_mechanisms]] — the wider family of attentional processes
- [[spatial_attention]] — selection of locations in space
- [[exogenous_attention]] — stimulus-driven orienting
- [[endogenous_attention]] — goal-directed orienting
- [[attentional_control]] — the executive component of the network model

## References
- Posner, M. I. (1980). Orienting of attention. Quarterly Journal of Experimental Psychology, 32, 3-25.
- Posner, M. I., & Petersen, S. E. (1990). The attention system of the human brain. Annual Review of Neuroscience, 13, 25-42.
- Feldman, H., & Friston, K. J. (2010). Attention, uncertainty, and free-energy. Frontiers in Human Neuroscience, 4, 215.
