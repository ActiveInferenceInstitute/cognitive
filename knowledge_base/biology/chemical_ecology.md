---
title: Chemical Ecology
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - chemical_ecology
  - pheromones
  - semiochemicals
  - communication
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/biology/ant_communication_systems]]"
      - "[[knowledge_base/biology/bee_communication_systems]]"
      - "[[knowledge_base/cognitive/stigmergic_coordination]]"
---

# Chemical Ecology

## Overview
Chemical ecology studies how organisms use chemical signals and cues to mediate interactions with each other and with their environment. Its core objects are semiochemicals, molecules that carry information between organisms, and the secondary metabolites that plants, microbes, and animals use for defense, attraction, and recognition.

## Mechanism
Semiochemicals divide into pheromones, which act between members of the same species, and allelochemicals, which act between species. Allelochemicals are classified by who benefits: allomones benefit the emitter (a defensive toxin), kairomones benefit the receiver (a prey odor used by a predator), and synomones benefit both (floral scent attracting a pollinator). The first pheromone to be chemically identified was bombykol from the silkworm moth Bombyx mori (Butenandt and colleagues, 1959), and the term pheromone was introduced by Karlson and Lüscher the same year.

In social insects the system is highly developed. Ants use trail pheromones that are deposited, reinforced by traffic, and lost by evaporation; alarm pheromones that recruit defenders; and cuticular hydrocarbon profiles that let nestmates distinguish colony members from intruders. Signal properties such as volatility, persistence, and diffusion set the spatial and temporal range of a message, so the chemistry itself determines the dynamics of the communication channel. In plants, herbivore-induced volatiles can attract parasitoids of the attacking herbivore, and allelopathy (for example juglone release by black walnut) suppresses neighbors. Bacterial quorum sensing is a chemical mechanism of density-dependent coordination that is formally comparable to pheromone communication.

## Role in Active Inference
Chemical signals are the observation channel of the foraging agents described in [[knowledge_base/systems/ant_colony_optimization]] and in the Ant_Colony Thing, where observations include pheromone gradients and agents deposit pheromone as an action. A deposited pheromone field is external state that persists, decays, and is read by other agents, so it acts as shared memory. Describing this as inference is a modeling choice: the field itself performs no inference, but agents can treat it as a noisy cue about food location and update beliefs accordingly. The decay constant then fixes how much weight old information should carry.

## Related concepts
- [[knowledge_base/biology/ant_communication_systems]] — the concrete signals used by ants
- [[knowledge_base/biology/bee_communication_systems]] — pheromones and dance communication in bees
- [[knowledge_base/cognitive/stigmergic_coordination]] — coordination through persistent environmental traces
- [[knowledge_base/biology/pollination_biology]] — floral scent as a synomone mediating plant-pollinator contact
- [[knowledge_base/biology/ecological_networks]] — interaction webs that chemical signals structure

## References
- Karlson, P., and Lüscher, M. (1959). Pheromones: a new term for a class of biologically active substances. Nature.
