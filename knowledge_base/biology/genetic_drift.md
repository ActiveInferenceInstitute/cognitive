---
title: Genetic Drift
description: Random changes in allele frequencies caused by finite population size, with the Wright-Fisher model, fixation probability, founder and bottleneck effects, the neutral theory, and its weak relation to Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - population_genetics
  - evolution
  - stochastic_processes
semantic_relations:
  - type: relates
    links:
      - "[[population_genetics]]"
      - "[[natural_selection]]"
      - "[[evolutionary_dynamics]]"
      - "[[genetic_diversity]]"
      - "[[speciation]]"
      - "[[stochastic_processes]]"
---

# Genetic Drift

## Overview
Genetic drift is the change in allele frequencies from one generation to the next caused by random sampling in a finite population. Not every individual reproduces equally by chance, and the gametes that form the next generation are a sample of the parental gene pool. Drift acts in every real population, and is strongest when populations are small.

## Quantitative picture
In the Wright-Fisher model, a population of $N$ diploid individuals produces the next generation by sampling $2N$ gene copies with replacement. For an allele at frequency $p$, the sampling variance of the next-generation frequency is
$$\operatorname{Var}(\Delta p)=\frac{p(1-p)}{2N},$$
so drift is weaker in larger populations. Several results follow:
- Drift lowers heterozygosity over time and eventually fixes one allele and loses the others.
- For a neutral allele, the probability of eventual fixation equals its current frequency.
- Fixation of a new neutral mutation takes a time on the order of $4N_e$ generations, where $N_e$ is the effective population size.
- **Founder effects and bottlenecks** are cases where a small number of individuals found or survive to re-found a population, so frequencies shift sharply and diversity declines (see [[genetic_diversity]]).

## Role in evolution
Drift competes with [[natural_selection]]: an allele with selection coefficient $s$ behaves as effectively neutral when $|s|$ is much smaller than about $1/N_e$. Kimura's neutral theory (Kimura, 1968) proposed that most molecular variation within and between species reflects drift acting on selectively neutral mutations, a view that has been influential though not universally accepted for all classes of sites. Drift also contributes to divergence among isolated populations and thereby to [[speciation]].

## Relation to Active Inference
The relation is slight. Drift is a stochastic process described by standard population-genetic models, and it is not derived from the free energy principle. Some theoretical work casts selection as a form of Bayesian model selection, in which random effects such as drift play the role of noise in the dynamics, but this is an interpretation, and drift itself is treated by the stochastic models above (see [[stochastic_processes]]).

## Related concepts
- [[population_genetics]] — the theoretical framework where drift is formalised
- [[natural_selection]] — the non-random force contrasted with drift
- [[evolutionary_dynamics]] — mathematical treatment of change in populations
- [[genetic_diversity]] — the variation that drift erodes
- [[speciation]] — divergence of isolated populations

## References
- Kimura, M. (1968). Evolutionary rate at the molecular level. Nature, 217, 624-626.
- Gillespie, J. H. (2004). Population Genetics: A Concise Guide (2nd ed.). Johns Hopkins University Press.
- Hartl, D. L., & Clark, A. G. (2007). Principles of Population Genetics (4th ed.). Sinauer Associates.
