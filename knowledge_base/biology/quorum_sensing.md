---
title: Quorum Sensing
description: "Collective decision-making in which a group commits once enough individuals have independently chosen an option, from bacterial autoinducers to ant and honeybee nest-site selection."
type: biology
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - collective-behavior
  - social-insects
  - decision-making
semantic_relations:
  - type: relates
    links:
      - "[[swarm_intelligence]]"
      - "[[collective_behavior_ants]]"
      - "[[social_insect_cognition]]"
---

# Quorum Sensing

## Overview
Quorum sensing is a mechanism by which a group changes behavior once the number or density of participants crosses a threshold. The term originated in microbiology, where bacteria release signalling molecules called autoinducers and switch on collective behavior, such as luminescence in *Vibrio fischeri*, when concentration reflects a sufficient population. In the social-insect literature the same logic is applied to groups of animals that estimate how many others support an option; this page covers mainly that animal sense, with the bacterial case as the origin of the name.

## Mechanism in social insects
In nest-site selection by the ant *Temnothorax albipennis*, scouts find candidate sites and recruit nestmates by tandem running, a slow one-to-one process. Pratt showed that scouts at a site respond to the encounter rate with other ants there: when the count exceeds a threshold (the quorum), they switch from tandem running to carrying nestmates, a much faster form of recruitment that commits the colony. In honeybee swarms, Seeley and Visscher found that scouts begin the process leading to departure once enough scouts are present at a site, with the quorum serving as the cue that a site has gained enough support to prepare the swarm for departure.

The key properties are non-linear and tolerant of individual error. Because the group responds to the number of independent sampling individuals rather than to one individual's opinion, a quorum amplifies a consistent majority signal and filters noise, and the threshold sets a trade-off: a low quorum is fast but error-prone, a high quorum is accurate but slow. Colonies adjust the quorum threshold with urgency, accepting lower accuracy under time pressure. Quorum responses are a collective counterpart of the individual [[response_thresholds]] that govern [[division_of_labor]], and they share the sigmoidal shape of a threshold function. The model is a mean-field description; real colonies show variance between individuals and the encounter-rate interpretation has been refined and debated.

## Role in Active Inference
A quorum can be read as evidence accumulation at the group level: individuals contribute noisy samples, and the group acts when posterior support for an option crosses a decision bound ([[evidence_accumulation]], [[decision_making]]). That reading is an analogy with sequential-sampling models; it has been proposed in discussions of [[active_inference_social_insects]] but is not a derivation from the free-energy principle.

## Related concepts
- [[bee_colony_organization]] — swarm and nest-site decisions in honeybees
- [[stigmergic_coordination]] — indirect coordination via the environment
- [[ant_colony_organization]] — recruitment systems in ants
- [[exploration_exploitation]] — speed-accuracy and commitment trade-offs

## References
- Seeley, T. D., & Visscher, P. K. (2004). Quorum sensing during nest-site selection by honeybee swarms. *Behavioral Ecology and Sociobiology*, 56.
- Pratt, S. C. (2005). Quorum sensing by encounter rates in the ant *Temnothorax albipennis*. *Behavioral Ecology*, 16.
