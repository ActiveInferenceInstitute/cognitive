---
title: Food Web Theory
description: The study of feeding relationships among species in ecological communities, covering network structure, energy flow, stability and complexity debates, keystone effects, and links to network science.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - ecology
  - networks
  - stability
semantic_relations:
  - type: relates
    links:
      - "[[trophic_levels]]"
      - "[[ecological_networks]]"
      - "[[ecosystem_stability]]"
      - "[[community_ecology]]"
      - "[[population_dynamics]]"
      - "[[network_science]]"
---

# Food Web Theory

## Overview
Food web theory studies who eats whom in an ecological community and what the resulting network implies for energy flow, population dynamics and stability. A food web represents species, or groups of species, as nodes and feeding links as directed edges. Elton (1927) introduced food cycles in animal ecology, and Lindeman (1942) recast communities in terms of energy transfer between trophic levels (see [[trophic_levels]]).

## Structure and properties
- **Measures.** Typical descriptors are the number of species $S$, the number of links $L$, and connectance, the fraction of possible links realised, $C = L/S^2$. Others include chain length, the proportion of top and basal species, and degree distributions.
- **Empirical webs.** Observed webs are hard to compile consistently because the resolution of taxa and the effort of sampling vary, which affects estimated properties.
- **Keystone effects.** Paine (1966) showed that removing a predator from an intertidal community lowered diversity, an experiment that underlies the keystone species concept. Trophic cascades describe indirect effects of predators on lower levels.

## Stability and complexity
May (1972) analysed randomly assembled communities with linear stability theory and found that a system becomes unstable as the number of species, connectance or interaction strength rises beyond a threshold, which contradicted the view that complexity brings stability. Later work showed that real webs are not random: they have structured link patterns, weak interactions, compartments and non-random feeding hierarchies that can support persistence (Pimm, 1982). The relation between diversity and stability depends on how both are defined (see [[ecosystem_stability]]).

## Relation to Active Inference
The link is thin. Food webs are studied with ecological and network methods (see [[network_science]]), and they are not derived from the free energy principle. Theoretical work in Active Inference sometimes describes ecosystems as nested systems of interacting agents, but there is no accepted food web result in that framework, and the connection should be treated as speculative.

## Related concepts
- [[trophic_levels]] — the tiered organisation of feeding
- [[ecological_networks]] — the general class of species-interaction networks
- [[ecosystem_stability]] — persistence and resilience of communities
- [[community_ecology]] — the field that studies assemblages of species
- [[population_dynamics]] — predator-prey and consumer-resource models

## References
- Elton, C. S. (1927). Animal Ecology. Sidgwick and Jackson.
- Lindeman, R. L. (1942). The trophic-dynamic aspect of ecology. Ecology, 23, 399-417.
- Paine, R. T. (1966). Food web complexity and species diversity. The American Naturalist, 100, 65-75.
- May, R. M. (1972). Will a large complex system be stable? Nature, 238, 413-414.
- Pimm, S. L. (1982). Food Webs. Chapman and Hall.
