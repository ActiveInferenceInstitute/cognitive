---
title: Adaptive Dynamics
description: A mathematical framework for the long-term evolution of continuous traits under frequency-dependent selection, built on invasion fitness, the canonical equation and the classification of evolutionarily singular strategies.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - evolutionary_theory
  - invasion_fitness
  - population_dynamics
semantic_relations:
  - type: relates
    links:
      - "[[evolutionary_dynamics]]"
      - "[[evolutionary_game_theory]]"
      - "[[fitness_landscapes]]"
      - "[[population_dynamics]]"
      - "[[natural_selection]]"
---

# Adaptive Dynamics

## Overview
Adaptive dynamics is a body of theory for modelling how quantitative traits evolve when fitness depends on what other individuals in the population are doing. It combines ecological population dynamics with an evolutionary process of rare, small mutations, and it is used to study questions such as the evolution of dispersal, virulence, resource use and speciation (see also [[evolutionary_dynamics]]).

## Invasion fitness
The central quantity is the invasion fitness $s(y,x)$: the long-term per-capita growth rate of a rare mutant with trait value $y$ in a resident population with trait $x$ at its ecological attractor. A mutant can invade if $s(y,x)>0$. Because the environment of the mutant is set by the resident, fitness is frequency dependent, and unlike a static [[fitness_landscapes]] picture the landscape changes as the population moves across it. The method assumes that mutations are rare and small, so that the resident population reaches its equilibrium before the next mutant appears, and that reproduction is clonal or approximated as such.

## Canonical equation and singular strategies
Under these assumptions, the mean trait follows the canonical equation of Dieckmann and Law (1996):
$$\frac{dx}{dt}=\tfrac12\,\mu(x)\,\sigma^2(x)\,\hat N(x)\,\left.\frac{\partial s(y,x)}{\partial y}\right|_{y=x},$$
where $\mu$ is the per-capita mutation rate (mutation probability per birth times the birth rate), $\sigma^2$ the variance of mutational step size and $\hat N$ the equilibrium population density. The derivative is the selection gradient. Trait values where it vanishes are evolutionarily singular strategies, which are classified by two second-order conditions: whether a strategy cannot be invaded by neighbouring mutants (evolutionary stability) and whether the population approaches it from nearby values (convergence stability). A strategy that is convergence stable but invadable produces evolutionary branching, in which a monomorphic population splits into two coexisting lineages (Geritz et al., 1998). Pairwise invasibility plots display the sign of $s(y,x)$ to read off these properties.

## Relation to Active Inference
There is no direct derivation. Evolutionary dynamics have been related to Bayesian inference, with selection read as updating a distribution over phenotypes by relative fitness, as in discrete replicator dynamics, and the free energy principle has been proposed as a description of the persistence of self-organising systems. These are conceptual correspondences under active discussion, and adaptive dynamics itself, which is mainly concerned with ecological feedback on selection, is not usually framed in those terms.

## Related concepts
- [[evolutionary_dynamics]] — the wider study of change in populations over time
- [[evolutionary_game_theory]] — frequency-dependent payoffs and stable strategies
- [[fitness_landscapes]] — the static picture that adaptive dynamics generalises
- [[population_dynamics]] — the ecological layer that determines invasion fitness
- [[natural_selection]] — the process driving the trait change

## References
- Metz, J. A. J., Geritz, S. A. H., Meszéna, G., Jacobs, F. J. A., & van Heerwaarden, J. S. (1996). Adaptive dynamics, a geometrical study of the consequences of nearly faithful reproduction. In S. J. van Strien & S. M. Verduyn Lunel (Eds.), Stochastic and Spatial Structures of Dynamical Systems (pp. 183-231). North-Holland.
- Dieckmann, U., & Law, R. (1996). The dynamical theory of coevolution: A derivation from stochastic ecological processes. Journal of Mathematical Biology, 34, 579-612.
- Geritz, S. A. H., Kisdi, E., Meszéna, G., & Metz, J. A. J. (1998). Evolutionarily singular strategies and the adaptive growth and branching of the evolutionary tree. Evolutionary Ecology, 12, 35-57.
