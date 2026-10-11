---
title: Hardy-Weinberg Equilibrium
description: The principle that allele and genotype frequencies in a large randomly mating population stay constant across generations when evolutionary forces are absent, serving as the null model of population genetics.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - population_genetics
  - null_model
  - allele_frequencies
semantic_relations:
  - type: relates
    links:
      - "[[population_genetics]]"
      - "[[genetic_drift]]"
      - "[[natural_selection]]"
      - "[[genetic_diversity]]"
      - "[[evolutionary_dynamics]]"
---

# Hardy-Weinberg Equilibrium

## Overview
Hardy-Weinberg equilibrium (HWE) states that, in the absence of evolutionary forces, allele frequencies in a population do not change from generation to generation, and genotype frequencies settle at fixed proportions determined by those allele frequencies. The principle was published independently in 1908 by G. H. Hardy and by Wilhelm Weinberg. It is the baseline against which deviations, and so evolutionary processes, are detected.

## Statement
Consider an autosomal locus with two alleles $A$ and $a$ at frequencies $p$ and $q=1-p$. After one generation of random mating the genotype frequencies are
$$f(AA)=p^2,\qquad f(Aa)=2pq,\qquad f(aa)=q^2,$$
and these sum to $(p+q)^2=1$. The allele frequencies in the next generation are again $p$ and $q$, so the proportions persist. For $k$ alleles with frequencies $p_i$, homozygotes occur at $p_i^2$ and heterozygotes for alleles $i$ and $j$ at $2p_ip_j$.

## Assumptions
The principle holds for a diploid, sexually reproducing population with random mating, no selection, no mutation, no migration, and a population large enough that drift is negligible, with non-overlapping generations. Violating each assumption has a recognisable signature: selection or mutation shifts allele frequencies, small size produces drift (see [[genetic_drift]]), and inbreeding or population structure produces a deficit of heterozygotes.

## Use
Observed genotype counts can be compared with expectations from the estimated allele frequencies with a chi-square test, which for a two-allele locus has one degree of freedom. Deviations can indicate non-random mating, hidden population structure, selection or genotyping error, and the test is routinely used for quality control in genetic association studies. A population in HWE is not necessarily at an evolutionary standstill, because a locus can fit the proportions while other loci change.

## Relation to Active Inference
There is no direct link. The principle belongs to classical population genetics. Work that treats evolution as a form of inference over environmental regularities (see [[free_energy_principle]]) takes allele-frequency dynamics as its subject, and HWE then serves only as the neutral reference case for such models.

## Related concepts
- [[population_genetics]] — the field in which the principle is a basic result
- [[genetic_drift]] — random change in allele frequencies in finite populations
- [[natural_selection]] — a force that moves frequencies away from equilibrium
- [[genetic_diversity]] — variation that the principle helps describe
- [[genetics]] — heredity and variation in general

## References
- Hardy, G. H. (1908). Mendelian proportions in a mixed population. Science, 28, 49-50.
- Hartl, D. L., & Clark, A. G. (2007). Principles of Population Genetics (4th ed.). Sinauer Associates.
- Gillespie, J. H. (2004). Population Genetics: A Concise Guide (2nd ed.). Johns Hopkins University Press.
