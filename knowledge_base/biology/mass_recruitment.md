---
title: Mass Recruitment
description: Recruitment of many nestmates to a resource through pheromone trails laid by successful foragers, and the positive and negative feedback loops that make the process self-organising.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - ants
  - recruitment
  - self_organization
semantic_relations:
  - type: relates
    links:
      - "[[trail_pheromones]]"
      - "[[stigmergic_coordination]]"
      - "[[tandem_running]]"
      - "[[collective_behavior_ants]]"
      - "[[self_organization]]"
---

# Mass Recruitment

## Overview
Mass recruitment is a mode of recruitment in which a single forager can mobilise many nestmates at once. In ants it usually works through a pheromone trail laid on the ground between the food and the nest. It contrasts with one-to-one recruitment, such as [[tandem_running]], and with solitary foraging, where no information is passed on.

## Mechanism
A forager that finds food and returns to the nest deposits pheromone, often with its gaster. Nestmates leaving the nest encounter the trail and tend to follow it, and those that find food reinforce it on their return. Two opposing processes shape the outcome:
- **Positive feedback.** More ants on a trail means more deposition, which attracts still more ants.
- **Negative feedback.** The pheromone evaporates, and a source that runs out stops being reinforced, so unused trails fade.

Following is probabilistic and the response to pheromone concentration is nonlinear, so small initial differences between trails can be amplified into a collective choice. In experiments with the Argentine ant *Linepithema humile* on a bridge with two branches of different length, colonies tend to converge on the shorter route because ants on the shorter branch return sooner and reinforce it earlier (Goss et al., 1989). Models of path selection by *Lasius niger*, which include the ants' U-turns on the trail, reproduce how a colony chooses between routes and show how early random differences are amplified (Beckers et al., 1992). Because the outcome depends on such amplification, mass recruitment is adaptive on average but not guaranteed to be optimal.

## Relation to Active Inference
The relationship to Active Inference is an analogy. The trail is an external memory that changes the sensory input of later ants, which makes mass recruitment a standard example of [[stigmergic_coordination]]. The same mechanism inspired optimisation algorithms (see [[ant_colony_optimization]]). Recasting colony foraging as inference is an area of exploration rather than a settled result; see [[active_inference_social_insects]].

## Related concepts
- [[trail_pheromones]] — the chemical signals that carry the recruitment message
- [[stigmergic_coordination]] — coordination through traces left in the environment
- [[tandem_running]] — recruitment of a single nestmate at a time
- [[ant_colony_optimization]] — algorithms based on trail reinforcement
- [[self_organization]] — order arising from feedback among local interactions

## References
- Goss, S., Aron, S., Deneubourg, J. L., & Pasteels, J. M. (1989). Self-organized shortcuts in the Argentine ant. Naturwissenschaften, 76, 579-581.
- Beckers, R., Deneubourg, J. L., & Goss, S. (1992). Trails and U-turns in the selection of a path by the ant Lasius niger. Journal of Theoretical Biology, 159, 397-415.
- Hölldobler, B., & Wilson, E. O. (1990). The Ants. Belknap Press of Harvard University Press.
