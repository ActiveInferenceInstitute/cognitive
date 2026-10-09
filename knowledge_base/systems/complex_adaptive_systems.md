---
title: Complex Adaptive Systems
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - complex_adaptive_systems
  - adaptation
  - emergence
  - agent_based_modeling
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/systems/complex_systems]]"
      - "[[knowledge_base/systems/adaptive_systems]]"
      - "[[knowledge_base/systems/emergence]]"
      - "[[knowledge_base/free_energy_principle/systems/complex_adaptation]]"
---

# Complex Adaptive Systems

## Overview
A complex adaptive system (CAS) is a system of many interacting agents that modify their behavior in response to experience, producing aggregate patterns that no agent intends or controls. The term is associated with the Santa Fe Institute and with John Holland, who distinguished CAS from complex systems in general by the presence of adaptation: the agents and the system as a whole change their rules over time.

## Mechanism
Holland (1995) listed seven features. Four are properties: aggregation (agents form higher-level groups), nonlinearity, flows (of resources and information, with multiplier and recycling effects), and diversity. Three are mechanisms: tagging (labels that allow agents to select and form aggregates), internal models (agents anticipate by holding rules about their environment), and building blocks (reusable components recombined to make internal models). Levin (1998) characterized ecosystems and the biosphere as CAS by three properties: sustained diversity and individuality of components, localized interactions, and an autonomous process that uses the outcomes of local interactions to select a subset of those components for replication or enhancement.

Typical CAS include ant colonies, immune systems, brains, ecosystems, markets, and cities. They show several recurring dynamics: path dependence, in which early events bias later outcomes; attractors and regime shifts rather than smooth equilibria; power-law distributions in some quantities; and robustness to some perturbations together with fragility to others. Because there is no closed-form description, the main method is agent-based modeling, in which heterogeneous agents are simulated and macroscopic statistics are measured. Control is generally by shaping incentives, constraints, and information flows rather than by direct command, and interventions often have unintended consequences.

## Role in Active Inference
The fit is strong in vocabulary and weaker in proof. Holland's internal models correspond to the generative models of Active Inference agents, and adaptation to updating those models through experience. The free-energy principle also supplies a formal account of why systems with Markov-blanket structure remain in characteristic states, and so a way to frame self-organization in a CAS. The extension from a single bounded system to a population of interacting adaptive agents is the contribution of multi-agent work, and it remains less mature than the single-agent theory; claims that the free-energy principle explains CAS in general go beyond what has been demonstrated. In this repository, the Ant_Colony Thing is a small CAS: heterogeneous-state agents, local pheromone interaction, and emergent trails.

## Related concepts
- [[knowledge_base/systems/complex_systems]] — the broader theory of interacting nonlinear components
- [[knowledge_base/systems/adaptive_systems]] — adaptation and learning in systems
- [[knowledge_base/systems/emergence]] — higher-level patterns from lower-level interaction
- [[knowledge_base/free_energy_principle/systems/self_organization]] — the free-energy view of spontaneous order
- [[knowledge_base/free_energy_principle/systems/complex_adaptation]] — complex adaptation under the free-energy principle
- [[knowledge_base/ontology/systems_ontology]] — the ontology page that links to this concept
- [[knowledge_base/cognitive/swarm_intelligence]] — a canonical biological CAS

## References
- Holland, J. H. (1995). Hidden Order: How Adaptation Builds Complexity.
- Levin, S. A. (1998). Ecosystems and the biosphere as complex adaptive systems. Ecosystems.
