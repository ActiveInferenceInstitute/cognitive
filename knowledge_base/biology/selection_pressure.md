---
title: Selection Pressure
description: The strength with which environmental factors favour some heritable traits over others in a population, how it is measured, the main modes of selection, and analogies with Bayesian inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - evolution
  - natural_selection
  - population_biology
semantic_relations:
  - type: relates
    links:
      - "[[natural_selection]]"
      - "[[evolutionary_biology]]"
      - "[[fitness_landscapes]]"
      - "[[population_genetics]]"
      - "[[phenotypic_plasticity]]"
---

# Selection Pressure

## Overview
Selection pressure is any environmental factor that causes individuals with certain heritable traits to survive or reproduce at different rates from others. It is a shorthand for the cause of differential fitness in [[natural_selection]]. Sources include predation, climate, food supply, pathogens, competition for mates, and human interventions such as antibiotics. The term is metaphorical: environments do not push populations in a direction, they filter variation that already exists.

## Measurement and modes
Selection is quantified as the covariance between a trait and relative fitness. Lande and Arnold (1983) formalised the measurement of selection on correlated characters using regression, separating direct selection on each trait from selection transmitted through correlations. In quantitative genetics the response to selection is often summarised by the breeder's equation $R = h^2 S$, where $S$ is the selection differential, $h^2$ is narrow-sense heritability and $R$ is the change in the population mean in the next generation.

Three modes are usually distinguished:
- **Directional**: one extreme of a trait is favoured, shifting the mean.
- **Stabilising**: intermediate values are favoured, reducing variance.
- **Disruptive**: both extremes are favoured over intermediates.

Field studies show that the strength and even the direction of selection can vary between years. In Darwin's finches on Daphne Major, droughts and rainfall changes altered which beak sizes were favoured over decades (Grant and Grant, 2002). Endler (1986) reviewed methods and examples of detecting selection in natural populations. Selection pressure can be altered by the organisms themselves through habitat choice and [[niche_construction]].

## Relation to Active Inference
Several authors have proposed that selection can be described as a form of inference, with a population or lineage accumulating information about its environment. Campbell (2016) argued that Darwinian processes are instances of Bayesian updating, and the Free Energy Principle is sometimes applied to evolutionary dynamics, treating phenotypes as embodying a model of their niche. These are formal analogies and active theoretical programmes, not settled results. Standard population genetics does not require them.

## Related concepts
- [[natural_selection]] — the process driven by selection pressure
- [[fitness_landscapes]] — a geometric picture of how pressures shape adaptation
- [[population_genetics]] — mathematical treatment of allele frequency change
- [[phenotypic_plasticity]] — traits that respond to the environment within a lifetime
- [[niche_construction]] — organisms modifying the pressures they experience
- [[evolutionary_biology]] — the broader field

## References
- Lande, R., & Arnold, S. J. (1983). The measurement of selection on correlated characters. Evolution, 37, 1210-1226.
- Endler, J. A. (1986). Natural Selection in the Wild. Princeton University Press.
- Grant, P. R., & Grant, B. R. (2002). Unpredictable evolution in a 30-year study of Darwin's finches. Science, 296, 707-711.
- Campbell, J. O. (2016). Universal Darwinism as a process of Bayesian inference. Frontiers in Systems Neuroscience, 10, 49.
