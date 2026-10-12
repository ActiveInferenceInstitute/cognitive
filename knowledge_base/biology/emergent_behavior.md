---
title: Emergent Behavior
description: Group-level patterns and capabilities that arise from local interactions among simpler components and are not specified in any individual's rules, with examples from ants, flocks and cells.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - collective_behavior
  - self_organization
  - complex_systems
semantic_relations:
  - type: relates
    links:
      - "[[emergence]]"
      - "[[self_organization]]"
      - "[[swarm_intelligence]]"
      - "[[stigmergic_coordination]]"
      - "[[collective_behavior]]"
      - "[[complex_adaptive_systems]]"
---

# Emergent Behavior

## Overview
Emergent behavior is a pattern at the level of a group that results from interactions among its members but is not found in, or directly programmed into, any single member. A foraging trail of ants, the coordinated turning of a bird flock and the formation of tissue patterns from cells are standard examples. The general concept is treated in [[emergence]]; this page concerns its behavioural and biological manifestations.

## Mechanisms
The studies of Camazine and colleagues (2001) identify recurring ingredients of such self-organised patterns.
- **Local interactions.** Individuals respond to nearby neighbours or to local cues, without a global view or a leader.
- **Positive feedback.** Amplification of initial fluctuations, for example pheromone reinforcement of a trail, drives pattern formation.
- **Negative feedback.** Saturation, evaporation or crowding limits growth and stabilises the pattern.
- **Randomness.** Noise lets the system explore alternatives and break symmetry.
- **Indirect coordination.** Modifications of a shared environment carry information between individuals, called stigmergy (see [[stigmergic_coordination]]).

In Reynolds' boids model (1987), three local rules suffice to produce flock-like motion: separation from close neighbours, alignment with neighbours' headings, and cohesion toward their centre. In ant colonies, task allocation and nest construction arise from encounter rates and cues, with no ant directing the colony (Gordon, 2010).

## Interpretation
The word is used in two senses. Weak emergence means that the pattern is derivable in principle from the components and rules but is practically unpredictable without simulation. Strong emergence, a stronger philosophical claim that the higher-level pattern is not derivable even in principle, is controversial and not required to explain the biological examples. Anderson's (1972) observation that new concepts apply at each level of organisation is often cited in this context. An emergent pattern is not necessarily optimal or adaptive, since selection acts on the individual rules.

## Relation to Active Inference
A research programme applies the free energy principle across scales, treating collections of agents with their own Markov blankets as a system with a blanket at the collective level, and modelling shared generative models in groups. Simulations show that coordination can arise when agents infer each other's states. The programme is active, but the claim that collective behavior in real animal groups follows this account is not established. The vault's pages on [[swarm_intelligence]] and [[collective_behavior]] cover empirical cases.

## Related concepts
- [[emergence]] — the general concept
- [[self_organization]] — order arising from internal dynamics without external control
- [[swarm_intelligence]] — collective problem solving by simple agents
- [[stigmergic_coordination]] — coordination through environmental traces
- [[complex_adaptive_systems]] — systems in which agents adapt as patterns emerge

## References
- Anderson, P. W. (1972). More is different. Science, 177, 393-396.
- Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. Computer Graphics, 21, 25-34.
- Camazine, S., Deneubourg, J.-L., Franks, N. R., Sneyd, J., Theraulaz, G., & Bonabeau, E. (2001). Self-Organization in Biological Systems. Princeton University Press.
- Gordon, D. M. (2010). Ant Encounters: Interaction Networks and Colony Behavior. Princeton University Press.
