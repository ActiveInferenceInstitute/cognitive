---
title: Population Dynamics
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - population_ecology
  - mathematical_biology
  - dynamical_systems
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/biology/ecological_dynamics]]"
      - "[[knowledge_base/mathematics/dynamical_systems]]"
      - "[[knowledge_base/biology/population_genetics]]"
wikidata: Q904564
---

# Population Dynamics

## Overview
Population dynamics describes how the number, density, and structure of individuals of a species change over time, as a result of births, deaths, immigration, and emigration. It supplies the basic mathematical models of ecology and links to evolutionary theory through demography.

## Mechanism
With constant per-capita rates, growth is exponential, dN/dt = rN. Resource limitation introduces density dependence, as in the logistic equation dN/dt = rN(1 - N/K), where K is the carrying capacity. The discrete-time logistic map shows that simple density dependence can produce oscillations and chaos when the growth rate is high (May 1976). Interacting species are described by the Lotka-Volterra predator-prey system, dx/dt = ax - bxy for prey and dy/dt = cxy - dy for predators, which generates neutral cycles in its simplest form and damped or limit-cycle behavior once prey self-limitation or functional responses are added. Competition models extend the same structure to several species and give conditions for coexistence.

Age and stage structure matter for real populations. The Leslie matrix projects an age-structured population forward by multiplying a state vector by a matrix of age-specific fecundities and survival probabilities; its dominant eigenvalue gives the asymptotic growth rate. At small population size, Allee effects (reduced per-capita growth at low density) and demographic stochasticity make extinction more likely than deterministic models predict. Metapopulation models (Levins) treat a population as a set of patches with colonization and local extinction.

Social insect colonies introduce a second level: the colony, not the worker, is the reproductive unit, so colony growth depends on worker production, foraging success, and the allocation between workers and reproductives. Population models of ants and bees therefore combine within-colony dynamics with between-colony competition and dispersal.

## Role in Active Inference
Population models are dynamical systems and can be integrated numerically; where a population is treated as an agent-based system, each individual can be given a generative model. The repository's [[knowledge_base/BioFirm/ecological_active_inference]] page discusses ecological applications. Strictly, the classical equations above are not derived from the free-energy principle; Active Inference enters when individual-level decision rules (foraging, reproduction, dispersal) are modeled as belief-driven and the population is then analyzed as an emergent aggregate. That is a modeling program, not an established equivalence.

## Related concepts
- [[knowledge_base/biology/ecological_dynamics]] — community-level dynamics and food webs
- [[knowledge_base/mathematics/dynamical_systems]] — stability and bifurcation analysis of these models
- [[knowledge_base/mathematics/ecological_systems]] — mathematical treatment of ecological systems
- [[knowledge_base/biology/population_genetics]] — allele-frequency dynamics within populations
- [[knowledge_base/biology/invasive_species]] — establishment and spread as population processes
- [[knowledge_base/systems/spatial_ecology]] — spatially structured populations

## References
- May, R. M. (1976). Simple mathematical models with very complicated dynamics. Nature.
