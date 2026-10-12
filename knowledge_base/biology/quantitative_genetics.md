---
title: Quantitative Genetics
description: The statistical study of continuously varying traits influenced by many genes and the environment, including heritability, the breeder's equation, genetic correlations, and its role in predicting evolutionary response.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - genetics
  - heritability
  - evolutionary_theory
semantic_relations:
  - type: relates
    links:
      - "[[population_genetics]]"
      - "[[genetics]]"
      - "[[evolutionary_biology]]"
      - "[[fitness_landscapes]]"
      - "[[bayesian_inference]]"
wikidata: Q550479
---

# Quantitative Genetics

## Overview
Quantitative genetics analyses traits such as height, milk yield, flowering time or body size that vary continuously and are influenced by many loci and by environment. Instead of following individual genes, it partitions observed variation into components using the resemblance between relatives. Fisher's 1918 paper showed that continuous variation is consistent with Mendelian inheritance at many loci, which reconciled the biometrical and Mendelian traditions and founded the field.

## Variance components and heritability
The phenotypic variance is decomposed as $V_P = V_G + V_E$, where $V_G$ is genetic and $V_E$ environmental variance. Genetic variance is itself divided into additive ($V_A$), dominance and epistatic parts. Narrow-sense heritability is
$$h^2 = \frac{V_A}{V_P},$$
the proportion of phenotypic variance attributable to additive effects. Heritability is a property of a trait in a particular population and environment. It does not measure how strongly genes determine a trait in an individual, and it can change when conditions or allele frequencies change.

## Prediction of response to selection
The breeder's equation, $R = h^2 S$, relates the response to selection $R$ to the selection differential $S$. For several traits, the multivariate form $\Delta\bar{z} = G\beta$ uses the additive genetic covariance matrix $G$ and the selection gradient $\beta$ (Lande's equation, with $\beta$ estimated by regression as in Lande and Arnold, 1983). Genetic correlations among traits, summarised in $G$, can bias or constrain the direction of evolution.

## Modern methods
Mixed models with pedigree or genomic relationship matrices estimate variance components, for example in the "animal model". Genome-wide association studies and polygenic scores relate many small-effect variants to trait variation, and genomic prediction is widely used in breeding. Estimates from association studies and from family studies sometimes differ, which is known as the "missing heritability" problem.

## Relation to Active Inference
There is a modest, partly speculative link. Some work casts natural selection as a form of Bayesian model selection, in which a population's genotypes encode implicit beliefs about its niche, and this has been explored within the free energy literature. Such treatments are interpretive and are not part of standard quantitative genetics, which rests on classical statistical models.

## Related concepts
- [[population_genetics]] — models allele-frequency change at single loci
- [[genetics]] — the general science of inheritance
- [[evolutionary_biology]] — the setting for predicting response to selection
- [[fitness_landscapes]] — geometric picture of selection on traits
- [[bayesian_inference]] — the statistical framework used in modern variance estimation

## References
- Fisher, R. A. (1918). The correlation between relatives on the supposition of Mendelian inheritance. Transactions of the Royal Society of Edinburgh, 52, 399-433.
- Lande, R., & Arnold, S. J. (1983). The measurement of selection on correlated characters. Evolution, 37, 1210-1226.
- Falconer, D. S., & Mackay, T. F. C. (1996). Introduction to Quantitative Genetics (4th ed.). Harlow: Longman.
- Lynch, M., & Walsh, B. (1998). Genetics and Analysis of Quantitative Traits. Sunderland, MA: Sinauer Associates.
