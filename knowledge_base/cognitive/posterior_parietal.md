---
title: Posterior Parietal Cortex
description: The posterior parietal cortex transforms sensory signals into spatial and motor representations for attention, reaching, and decision evidence, and has been interpreted as a site of precision control in Active Inference.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - parietal
  - attention
  - sensorimotor
semantic_relations:
  - type: relates
    links:
      - "[[parietal]]"
      - "[[attention_mechanisms]]"
      - "[[sensorimotor_coordination]]"
wikidata: Q3486606
---

# Posterior Parietal Cortex

## Overview
The posterior parietal cortex (PPC) is the part of the parietal lobe behind the primary somatosensory cortex, comprising the superior parietal lobule, the intraparietal sulcus, and the inferior parietal lobule ([[parietal]]). It sits between sensory areas and frontal motor and executive areas, and is classically described as a sensorimotor interface that represents space in coordinates useful for action. It also contributes to attention, working memory, and decision making.

## Functions
- **Intraparietal specialization in primates**: the lateral intraparietal area (LIP) is linked to saccade planning and attention; the parietal reach region and medial intraparietal area to reach planning; the anterior intraparietal area to grasping; and the ventral intraparietal area to multisensory space near the head. Andersen and Buneo (2002) proposed that PPC encodes intentions, meaning planned movements, in effector-specific maps.
- **Coordinate transformation**: neurons combine retinal position with eye, head, and limb position (gain fields), a mechanism for converting between reference frames ([[multisensory_integration]]).
- **Priority maps**: LIP activity is often described as a map of behavioural priority combining stimulus salience and goal relevance ([[spatial_attention]], [[attention_mechanisms]]).
- **Evidence accumulation**: LIP neuron firing rates ramp in proportion to accumulated motion evidence during random-dot decisions (Shadlen and Newsome, 2001), although whether LIP causally computes the decision is disputed ([[evidence_accumulation]]).
- **Consciousness and awareness**: damage causes hemispatial neglect, a deficit of awareness of one side of space, and parietal activity is prominent in studies of conscious access ([[consciousness_and_awareness]]).

## Role in Active Inference
Two proposals tie PPC to the framework. First, attention is the optimization of precision, which sets the gain of prediction errors; parietal priority-map activity fits this role, and Feldman and Friston (2010) formulated attention in this way. Second, in the motor system, PPC estimates of limb and target state supply the sensory predictions that descending commands specify, and reach planning can be framed as inference about a goal-directed trajectory ([[motor_networks]], [[proprioception]]). Neglect has been modelled as an asymmetry in precision or prior beliefs about space. The interpretations are consistent with physiology but are also covered by other formalisms such as optimal feedback control and biased-competition models, so the data do not single out Active Inference.

## Related concepts
- [[parietal]] — the lobe of which PPC is part
- [[inferior_parietal]] — lateral subdivision with language and social roles
- [[attention_mechanisms]] — attentional processes mediated by PPC
- [[sensorimotor_coordination]] — transformation of perception into action
- [[precision_weighting]] — gain control to which attention is mapped
- [[prefrontal]] — frontal partner in frontoparietal control networks

## References
- Andersen, R. A., & Buneo, C. A. (2002). Intentional maps in posterior parietal cortex. *Annual Review of Neuroscience*, 25, 189-220.
- Shadlen, M. N., & Newsome, W. T. (2001). Neural basis of a perceptual decision in the parietal cortex (area LIP) of the rhesus monkey. *Journal of Neurophysiology*, 86, 1916-1936.
- Feldman, H., & Friston, K. J. (2010). Attention, uncertainty, and free-energy. *Frontiers in Human Neuroscience*, 4, 215.
