---
title: Natural Selection
description: The process by which heritable variation in traits that affect survival or reproduction changes the composition of a population, covering Darwin's logic, the Price equation, modes of selection, and Bayesian readings.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - evolution
  - adaptation
  - population_genetics
semantic_relations:
  - type: relates
    links:
      - "[[genetic_drift]]"
      - "[[evolutionary_dynamics]]"
      - "[[fitness_landscapes]]"
      - "[[population_genetics]]"
      - "[[evolutionary_biology]]"
      - "[[phenotypic_plasticity]]"
wikidata: Q43478
---

# Natural Selection

## Overview
Natural selection is the differential survival and reproduction of individuals due to differences in heritable traits. Darwin (1859) argued that where individuals vary, where some variation is inherited, and where more offspring are produced than can survive, the traits associated with greater reproductive success become more common over generations. The process is the main known explanation for adaptation, the fit between organisms and their environments.

## Formal treatments
- **Breeder's equation.** The response to selection of a quantitative trait is $R=h^2S$, where $S$ is the selection differential and $h^2$ the narrow-sense heritability.
- **Price equation.** The change in the mean value $\bar z$ of a trait is
$$\bar w\,\Delta\bar z=\operatorname{Cov}(w,z)+\operatorname{E}(w\,\Delta z),$$
where $w$ is fitness. The covariance term is selection and the second term captures transmission bias. The equation is a general identity and can be applied at different levels, a point discussed in debates over the units of selection (Lewontin, 1970).
- **Modes.** Directional selection shifts a trait distribution, stabilising selection narrows it around an optimum, and disruptive selection favours both extremes. Selection can also be frequency-dependent, so that fitness depends on what others do (see [[evolutionary_dynamics]]).
- **Landscapes.** Evolutionary trajectories can be visualised as climbing peaks on a [[fitness_landscapes]] surface, with the caveat that landscapes change with the population and the environment.
- **Interaction with other forces.** Selection acts together with mutation, migration and [[genetic_drift]], and organisms with [[phenotypic_plasticity]] respond to environments within a lifetime as well.

## Relation to Active Inference
An active research programme interprets selection as a process akin to Bayesian model selection, in which phenotypes embody implicit models of their niche and persistence over generations scores those models, so that evolution appears to minimise free energy on a long timescale. This is a proposed unification with ongoing debate, not an established derivation, and the standard population-genetic account does not depend on it.

## Related concepts
- [[genetic_drift]] — random change competing with selection
- [[evolutionary_dynamics]] — mathematical description of population change
- [[fitness_landscapes]] — geometric picture of adaptation
- [[population_genetics]] — the quantitative theory of allele frequencies
- [[evolutionary_biology]] — the field in which selection is central

## References
- Darwin, C. (1859). On the Origin of Species by Means of Natural Selection. John Murray.
- Lewontin, R. C. (1970). The units of selection. Annual Review of Ecology and Systematics, 1, 1-18.
- Futuyma, D. J., & Kirkpatrick, M. (2017). Evolution (4th ed.). Sinauer Associates.
