---
title: Trail Pheromones
description: Chemical marks laid by ants and some other insects to guide nestmates to food or new nest sites, covering the positive-feedback mechanism, evaporation, shortest-path selection, and links to stigmergy and ant colony optimization.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - chemical_communication
  - social_insects
  - self_organization
semantic_relations:
  - type: relates
    links:
      - "[[alarm_pheromones]]"
      - "[[stigmergic_coordination]]"
      - "[[ant_colony_optimization]]"
      - "[[ant_communication_systems]]"
      - "[[trail_following]]"
      - "[[swarm_intelligence]]"
---

# Trail Pheromones

## Overview
Trail pheromones are chemical substances deposited on the ground by foraging insects, most thoroughly studied in ants, that other workers detect and follow. A worker that finds food returns to the nest while marking the route, and nestmates that encounter the mark are likely to follow it, find the food, and reinforce the trail on their return. Wilson's work on fire ants in the early 1960s established that such trails are chemical in nature and can guide workers over distance.

## Mechanism
- **Positive feedback.** More ants on a trail means more pheromone, which attracts more ants. Recruitment is therefore self-reinforcing, and a good food source quickly draws a large flow of workers.
- **Evaporation and decay.** Pheromone is volatile and degrades, so unreinforced trails fade. This provides negative feedback that lets a colony abandon exhausted sources.
- **Shortest-path selection.** In a binary-bridge experiment with Argentine ants, the colony usually came to use the shorter of two branches, because ants on the shorter branch completed round trips sooner and reinforced it sooner (Goss et al., 1989; Deneubourg et al., 1990). Chance fluctuations early on can lock in either branch when branch lengths are equal.
- **Variation across species.** Trail chemistry, persistence and the degree to which individuals follow trails differ among species, and some species also use visual or memory cues (see [[trail_following]]).

## Stigmergy and computation
Trail laying is a standard example of stigmergy, a term introduced by Grassé, in which the traces left in a shared environment coordinate the actions of individuals without direct communication (see [[stigmergic_coordination]]). The mechanism inspired [[ant_colony_optimization]], a family of algorithms for combinatorial problems.

## Relation to Active Inference
The pheromone field acts as an external memory shared by the colony, and models of foraging can treat it as part of the environment's state that agents sense and modify. Describing the colony as performing collective inference through such a field is an active modelling programme (see [[active_inference_social_insects]]), and it is an interpretive framework, not a finding about the chemistry or behaviour above.

## Related concepts
- [[alarm_pheromones]] — short-lived chemical signals of danger
- [[stigmergic_coordination]] — coordination through environmental traces
- [[ant_colony_optimization]] — algorithms inspired by trail dynamics
- [[ant_communication_systems]] — the wider set of ant signals
- [[swarm_intelligence]] — collective problem solving without central control

## References
- Goss, S., Aron, S., Deneubourg, J. L., & Pasteels, J. M. (1989). Self-organized shortcuts in the Argentine ant. Naturwissenschaften, 76, 579-581.
- Deneubourg, J. L., Aron, S., Goss, S., & Pasteels, J. M. (1990). The self-organizing exploratory pattern of the Argentine ant. Journal of Insect Behavior, 3, 159-168.
- Hölldobler, B., & Wilson, E. O. (1990). The Ants. Belknap Press of Harvard University Press.
