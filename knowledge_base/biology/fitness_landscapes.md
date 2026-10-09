---
title: Fitness Landscapes
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - evolution
  - fitness
  - epistasis
  - optimization
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/biology/evolutionary_biology]]"
      - "[[knowledge_base/biology/evolutionary_dynamics]]"
      - "[[knowledge_base/mathematics/evolutionary_algorithms]]"
---

# Fitness Landscapes

## Overview
A fitness landscape maps each genotype (or phenotype, or allele-frequency vector) to a fitness value, so that evolution can be pictured as movement over a surface with peaks, ridges, and valleys. Sewall Wright introduced the idea in 1932 to discuss how populations might move between adaptive peaks, and it has since become a general tool in evolutionary theory, molecular evolution, and optimization.

## Mechanism
In the genotype-space version, genotypes are vertices of a hypercube in which neighbors differ by one mutation, and fitness is a function on those vertices. The shape of the landscape depends on epistasis, the dependence of a mutation's effect on the genetic background. With no epistasis the landscape is smooth with a single peak; with strong epistasis it is rugged with many local optima, and some mutation paths become inaccessible because they require passing through a fitness decrease (sign epistasis). Kauffman's NK model (1993) makes ruggedness tunable: N genes each interact with K others, and increasing K increases the number of local peaks and shortens adaptive walks.

Under strong selection and weak mutation, a population performs an adaptive walk that climbs uphill until it reaches a local optimum. Drift, recombination, large populations, or changing environments allow other dynamics: crossing valleys, moving along neutral networks of equal-fitness genotypes, or tracking a landscape that itself moves (the fitness landscape depends on other species in coevolution). Fisher's geometric model offers a phenotypic alternative, in which fitness depends on distance from an optimum in a multidimensional trait space. A caveat is that a single static surface is a simplification: fitness is frequency-dependent and environment-dependent in many real systems.

## Role in Active Inference
The free-energy literature proposes that fitness can be identified with log model evidence, so that natural selection performs a kind of Bayesian model selection over phenotypes; under this reading a fitness landscape corresponds to a surface of negative surprise over phenotypes. This is a proposal from within the framework and remains debated; it is not an empirical consensus in evolutionary biology. In practical terms, the landscape metaphor is more reliably useful in this repository as an analogy for the variational free-energy surface that an agent descends during inference, and for the search spaces of the evolutionary algorithms, where ruggedness determines whether hill-climbing stalls in local optima.

## Related concepts
- [[knowledge_base/biology/evolutionary_biology]] — the wider theory the landscape metaphor serves
- [[knowledge_base/biology/evolutionary_dynamics]] — formal dynamics of populations on landscapes
- [[knowledge_base/biology/population_genetics]] — drift, selection, and mutation on allele frequencies
- [[knowledge_base/mathematics/evolutionary_algorithms]] — optimization methods whose behavior depends on landscape ruggedness
- [[knowledge_base/free_energy_principle/biology/evolution]] — the free-energy treatment of evolution
- [[knowledge_base/mathematics/stochastic_optimization]] — escaping local optima with noise

## References
- Wright, S. (1932). The roles of mutation, inbreeding, crossbreeding and selection in evolution. Proceedings of the Sixth International Congress of Genetics.
- Kauffman, S. A. (1993). The Origins of Order: Self-Organization and Selection in Evolution.
