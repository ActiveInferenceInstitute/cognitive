---
title: Artificial Life
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - artificial_life
  - simulation
  - emergence
  - self_organization
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/biology/autopoiesis]]"
      - "[[knowledge_base/cognitive/swarm_intelligence]]"
      - "[[knowledge_base/mathematics/evolutionary_algorithms]]"
      - "[[knowledge_base/free_energy_principle/systems/self_organization]]"
wikidata: Q263847
---

# Artificial Life

## Overview
Artificial life (ALife) is the study of life-like behavior in synthetic media. Christopher Langton named the field at a 1987 workshop in Los Alamos, framing it as the study of life as it could be, not only life as we know it. Work is divided into soft ALife (software), hard ALife (robotics), and wet ALife (synthetic chemistry and biology).

## Mechanism
Typical systems implement local rules and let global structure emerge. Conway's Game of Life (1970) is a two-dimensional cellular automaton with simple birth and survival rules that produces gliders, oscillators, and computationally universal configurations; von Neumann's earlier self-reproducing automaton established that open-ended self-replication is logically possible with a universal constructor. Reynolds's Boids (1987) reproduces flocking from three local steering rules: separation, alignment, and cohesion. Evolutionary platforms such as Ray's Tierra (1991) and Avida (Adami and Ofria) let self-replicating programs mutate and compete for processor time and memory, and have been used to study the evolution of complexity and of cooperation. Sims's virtual creatures (1994) co-evolved body and controller under a physics simulator.

A central open problem is open-ended evolution: the continued production of novelty without a plateau, which most current systems fail to sustain. Another dispute concerns strong versus weak ALife, namely whether a sufficiently faithful simulation would be alive or only a model of life. Methodologically, ALife values synthesis as a route to understanding: if a minimal rule set reproduces a phenomenon, that sets a lower bound on what is needed to explain it, without showing it is the mechanism used by real organisms.

## Role in Active Inference
Active Inference has been applied to ALife-style problems. Friston, Levin, Sengupta and Pezzulo (2015) showed in simulation that cells minimizing free energy under a shared generative model can self-organize into a target morphology and regenerate after damage, and the free-energy principle itself is a claim about how self-organizing systems persist, which is the question autopoiesis poses. The Ant_Colony Thing in this repository is a small-scale instance of the ALife method: agents with simple generative models in a pheromone environment, evaluated for emergent trails. These are modeling demonstrations; they do not establish that biological morphogenesis or colonies actually run on such inference.

## Related concepts
- [[knowledge_base/biology/autopoiesis]] — a definition of life as self-production, often tested in ALife models
- [[knowledge_base/biology/morphogenesis]] — pattern formation that ALife and Active Inference models both simulate
- [[knowledge_base/cognitive/swarm_intelligence]] — collective behavior from local rules
- [[knowledge_base/mathematics/evolutionary_algorithms]] — optimization methods that share ALife's evolutionary machinery
- [[knowledge_base/systems/emergence]] — the general concept of higher-level pattern from local interaction
- [[knowledge_base/systems/complex_adaptive_systems]] — the theoretical frame for ALife populations

## References
- Langton, C. G. (Ed.) (1989). Artificial Life: Proceedings of an Interdisciplinary Workshop on the Synthesis and Simulation of Living Systems.
- Reynolds, C. W. (1987). Flocks, herds and schools: a distributed behavioral model. SIGGRAPH Computer Graphics.
- Friston, K., Levin, M., Sengupta, B., and Pezzulo, G. (2015). Knowing one's place: a free-energy approach to pattern regulation. Journal of the Royal Society Interface.
