---
title: Confusion Effect
description: The reduction in a predator's capture success when it attacks a group of similar moving prey, attributed to limits on how many targets the predator can track and single out.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - behavioral_ecology
  - predation
  - collective_behavior
  - perception
semantic_relations:
  - type: relates
    links:
      - "[[swarm_intelligence]]"
      - "[[collective_behavior]]"
      - "[[behavioral_biology]]"
      - "[[evolutionary_biology]]"
      - "[[selective_attention]]"
      - "[[emergent_behavior]]"
---

# Confusion Effect

## Overview
The confusion effect is the finding that a predator is less likely to capture a prey animal when that animal is part of a group, because the predator has difficulty picking out and targeting a single individual among many similar, moving ones. It is usually quantified as a decrease in the predator's attack-to-kill ratio with increasing group size. It is one of several proposed benefits of grouping, alongside dilution of individual risk, early detection of predators and coordinated evasion, and it is often cited as a reason why swarms, shoals and flocks form.

## Evidence
- Landeau and Terborgh (1986) studied how the visual oddity of an individual and the size of the group interact in predation, and discussed the confusion effect in that setting.
- Jeschke and Tollrian (2007) combined their own experiments on prey swarms of *Daphnia* with the earlier literature. In their experiments the larvae of *Aeshna* and *Chaoborus* were confused at high prey density, whereas *Libellula* and *Triturus* larvae were not. Across the 25 predator-prey systems they reviewed, predators were confused in 16. They suggest that tactile predators are generally susceptible, and that visual predators are susceptible mainly when their prey are highly agile.
- Ioannou, Guttal and Couzin (2012) paired a live predator, the bluegill sunfish, with computer-generated prey and found that virtual prey that behaved as a group were more likely to survive than prey that moved in other ways, so predation can select for coordinated group movement.

## Proposed mechanisms
Krakauer (1995) proposed a connectionist model in which the effect arises from a perceptual bottleneck: a predator has limited capacity to process the positions of many targets at once, so as the number of candidate targets rises the predator's chance of locking onto one falls. Later work has asked whether the effect depends on limited attention or on a simpler degradation of sensory information, and the answer may differ between predator species. The effect is therefore not universal; it depends on the sensory system of the predator, the similarity and agility of the prey, and their density.

## Relation to Active Inference
The connection is an interpretation and not an established result. A predator tracking a group can be described as an agent with a limited capacity to maintain beliefs about the position of several moving targets (see [[selective_attention]]); a crowd of similar targets makes the target of attack ambiguous, which is a loss of the precision with which any one target is represented. The description is a possible modelling choice, and the cited studies do not use the free energy principle. For the broader context of group behaviour see [[swarm_intelligence]] and [[collective_behavior]].

## References
- Landeau, L., & Terborgh, J. (1986). Oddity and the 'confusion effect' in predation. Animal Behaviour, 34(5), 1372-1380. doi:10.1016/s0003-3472(86)80208-1
- Krakauer, D. C. (1995). Groups confuse predators by exploiting perceptual bottlenecks: a connectionist model of the confusion effect. Behavioral Ecology and Sociobiology, 36(6), 421-429. doi:10.1007/bf00177338
- Jeschke, J. M., & Tollrian, R. (2007). Prey swarming: which predators become confused and why? Animal Behaviour, 74(3), 387-393. doi:10.1016/j.anbehav.2006.08.020
- Ioannou, C. C., Guttal, V., & Couzin, I. D. (2012). Predatory fish select for coordinated collective motion in virtual prey. Science, 337(6099), 1212-1215. doi:10.1126/science.1218919
