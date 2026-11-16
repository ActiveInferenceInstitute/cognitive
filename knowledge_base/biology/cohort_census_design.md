---
title: Cohort Census Design
description: Inclusive censuses and survival-conditioned endpoint sampling.
type: concept
status: draft
created: 2026-11-15
updated: 2026-11-15
tags:
  - evolution
  - experimental_design
semantic_relations:
  - type: relates
    links:
      - "[[natural_selection]]"
      - "[[population_genetics]]"
---

# Cohort Census Design

## Overview
A cohort census defines which individuals and reproductive events enter an analysis. Record the inclusion rule before estimating selection, drift, or transmission. An inclusive cohort retains zero-offspring parents and extinct lineages. An endpoint cohort conditioned on survival estimates a different distribution.

## Inclusive versus endpoint-conditioned cohorts
Let K be descendant count and S the event that a lineage survives to the endpoint. The inclusive law is P(K); the selected law is P(K | S) = P(K,S)/P(S), when P(S)>0. Replacing one with the other changes the estimand. Record losses, censoring and missing parentage separately from observed zero reproduction; a missing record is not a zero.

For an inclusive haploid census with N descendants sampled independently conditional on parental type frequency p, K | p is Binomial(N,p). If p varies across a shared environmental history, the unconditional law is a mixture. Its variance is N E[p(1-p)] + N² Var(p), not generally N p_bar(1-p_bar). Endpoint selection needs its own conditional likelihood; do not fit the inclusive binomial to selected survivors.

## Experiment-design reference card
- Unit: a declared ancestral cohort and its assigned descendants, with explicit ploidy and census times.
- Preparation: replicate matched initial frequencies; record shared environmental history and family identity.
- Measurements: linked parentage, reproduction including zeros, survival, censoring, and inclusion probabilities.
- Comparison: report inclusive and endpoint-conditioned estimates separately, with the conditioning event in each likelihood.
- Limitation: the Price identity accounts for realized change; causal selection requires a design that separates correlated causes.

## Relation to Active Inference
A survival-conditioned posterior and an inclusive population census have different conditioning information. A shared normalization formula alone does not equate their biological meaning.

## Related concepts
- [[genetic_drift]]
- [[natural_selection]]
- [[population_genetics]]

## References
- Price, G. R. (1970). Selection and covariance. Nature, 227, 520–521. https://doi.org/10.1038/227520a0

The sampling and variance formulas above follow directly from conditional probability and the law of total variance; this card proposes a design rather than reporting an experiment.
