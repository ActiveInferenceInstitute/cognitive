---
title: Species Richness
description: The number of species in a defined sample or area, how it is measured and compared, its relation to evenness and diversity indices, and its use as an indicator of ecosystem state.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - ecology
  - biodiversity
  - measurement
semantic_relations:
  - type: relates
    links:
      - "[[biodiversity]]"
      - "[[genetic_diversity]]"
      - "[[ecosystem_stability]]"
      - "[[conservation_biology]]"
      - "[[ecology]]"
      - "[[bioregional_state_space]]"
---

# Species Richness

## Overview
Species richness is the number of distinct species present in a given sample, habitat or region. It is the simplest measure of biodiversity and is easy to communicate, but it ignores how individuals are distributed among species. Two communities with the same richness can differ greatly if one is dominated by a single species and the other is balanced. The balance is described by evenness, and indices such as those of Shannon and Simpson combine richness and evenness (Magurran, 2004).

## Measurement
Richness depends strongly on sampling effort and area, so raw counts from samples of different size are not comparable. Standard approaches handle this:
- **Rarefaction** estimates the expected richness in a smaller common sample size, and extrapolation methods extend curves beyond the sample (Gotelli & Colwell, 2001).
- **Estimators** such as Chao1 use the number of rare species, those seen once or twice, to estimate how many species were missed.
- **Species-area relationship.** Richness commonly rises with area as $S=cA^{z}$, where $c$ and $z$ are fitted constants and $z$ is typically well below one.

Whittaker (1960) separated diversity at three scales: alpha diversity within a local site, beta diversity as the turnover among sites, and gamma diversity across a region. Richness is also affected by taxonomic knowledge and detectability, since cryptic or rare species are easily missed.

## Patterns and use
Richness tends to be higher towards the tropics, a pattern called the latitudinal gradient, and it varies with productivity, habitat heterogeneity and disturbance. It features in conservation planning and in monitoring programmes, and it is a common candidate indicator of ecosystem condition. Richness alone is a coarse indicator: a loss of specialist species can be hidden if invasive generalists replace them, so composition and function also matter.

## Relation to Active Inference
There is no direct link. In frameworks that describe a bioregion as a system with measurable states (see [[bioregional_state_space]]), richness serves as one observable among many that could inform an agent's model of ecosystem health. That is a way of using the measurement, not a theoretical property of Active Inference.

## Related concepts
- [[biodiversity]] — the broader concept of variety in life
- [[genetic_diversity]] — variation within species, a complementary level
- [[ecosystem_stability]] — properties of persistence that diversity may influence
- [[conservation_biology]] — the field that uses richness in assessments
- [[ecology]] — the study of organisms and their environment

## References
- Whittaker, R. H. (1960). Vegetation of the Siskiyou Mountains, Oregon and California. Ecological Monographs, 30, 279-338.
- Gotelli, N. J., & Colwell, R. K. (2001). Quantifying biodiversity: procedures and pitfalls in the measurement and comparison of species richness. Ecology Letters, 4, 379-391.
- Magurran, A. E. (2004). Measuring Biological Diversity. Blackwell Publishing.
