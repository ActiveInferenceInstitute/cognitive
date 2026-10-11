---
title: Genetic Diversity
description: Heritable variation among individuals within and between populations, its measures, the forces that generate and erode it, and its importance for adaptation and conservation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - genetics
  - conservation
  - population_genetics
semantic_relations:
  - type: relates
    links:
      - "[[population_genetics]]"
      - "[[genetic_drift]]"
      - "[[hardy_weinberg_equilibrium]]"
      - "[[species_richness]]"
      - "[[conservation_biology]]"
      - "[[natural_selection]]"
---

# Genetic Diversity

## Overview
Genetic diversity is the amount of heritable variation in the DNA of a population or species. It is the raw material on which natural selection acts, and one of the three levels of biodiversity commonly distinguished, together with species and ecosystem diversity. Populations with little variation have less capacity to adapt to changing conditions such as new pathogens or climate shifts.

## Measures
- **Expected heterozygosity** (gene diversity) at a locus with allele frequencies $p_i$ is $H_e=1-\sum_i p_i^2$, the probability that two alleles drawn at random differ (Nei, 1973).
- **Nucleotide diversity** is the average proportion of differing sites between two sequences sampled from the population.
- **Allelic richness** counts the alleles per locus, often adjusted for sample size, and is more sensitive than heterozygosity to the loss of rare alleles.
- **Effective population size** $N_e$ is the size of an idealised population that would lose variation at the same rate as the real one, and is usually well below the census size.

## Generation and loss
Mutation creates new alleles and recombination reshuffles them, while gene flow brings variation in from other populations. Variation is lost through genetic drift, which in a finite idealised population reduces expected heterozygosity by a factor of $1-1/(2N_e)$ per generation, and through bottlenecks and founder events. Inbreeding does not itself change allele frequencies but reduces heterozygosity and increases the expression of deleterious recessive alleles. Selection may either maintain variation, as in balancing selection, or erode it, as in directional selection.

## Conservation
Reduced diversity in small, isolated populations is associated with inbreeding depression and a lower adaptive potential, and so genetic monitoring is part of the management of threatened species (Frankham, 2005; Frankham et al., 2010). Interventions such as translocation to restore gene flow need to weigh the benefits against the risk of mixing locally adapted populations.

## Relation to Active Inference
No direct link exists. Evolutionary variants of the framework treat populations as sampling and testing models of their environments, in which diversity corresponds to a spread of candidate models, but this is an interpretive analogy.

## Related concepts
- [[population_genetics]] — the theoretical study of variation in populations
- [[genetic_drift]] — random loss of variation in finite populations
- [[hardy_weinberg_equilibrium]] — baseline relation between allele and genotype frequencies
- [[species_richness]] — diversity measured at the level of species
- [[conservation_biology]] — the applied field concerned with preserving variation

## References
- Nei, M. (1973). Analysis of gene diversity in subdivided populations. Proceedings of the National Academy of Sciences USA, 70, 3321-3323.
- Frankham, R. (2005). Genetics and extinction. Biological Conservation, 126, 131-140.
- Frankham, R., Ballou, J. D., & Briscoe, D. A. (2010). Introduction to Conservation Genetics (2nd ed.). Cambridge University Press.
