---
title: Multi-lag Stratified Preparations
description: Testing coarse-state predictive closure across preparation histories and lags.
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

# Multi-lag Stratified Preparations

## Overview
A coarse observable can hide predictive differences among fine states. Test closure using several preparations within the same coarse state and several observation lags. Agreement at one lag or in one stationary preparation is insufficient to establish autonomous coarse dynamics.

## Experiment-design reference card
- Declare a fine-state variable X, coarse observation Y = g(X), preparation protocol, sampling clock, and observation error model.
- Within each starting coarse state, prepare distinct fine-state mixtures. Stratify the analysis by preparation history rather than pooling those mixtures immediately.
- Measure coarse transition distributions at multiple predeclared lags, using comparable measurement conditions and held-out preparations.
- Compare P(Y at t+lag | Y at t, preparation) across preparations at each lag. Differences expose predictive dependence on unresolved state or history.
- Include a fine-state or history-dependent predictor as a comparator. Match sample size and uncertainty before interpreting a null difference.

## Finite-state criterion
For a row-stochastic fine transition matrix P, strong lumpability requires sum of P[x,x_next] over each destination block to be equal for all x in the same source block. Under this condition a coarse one-step transition Q exists for every initial preparation, and its powers predict all integer lags. Observed agreement at finitely many lags supports only those tested settings; it does not prove the universal criterion. Stationarity alone does not impose the block-sum equality.

## Relation to Active Inference
A coarse generative model can predict well within one preparation while failing after a history intervention. Evaluate predictive adequacy separately from claims of exact closure or mechanistic identity.

## Related concepts
- [[population_genetics]]
- [[evolutionary_dynamics]]
- [[cohort_census_design]]

## References
The block-sum criterion is stated explicitly here as a finite-matrix condition. The preparation protocol is a proposed diagnostic design, with no claimed experimental result.
