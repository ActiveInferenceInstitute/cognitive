---
title: Cue Reliability and Offspring Interventions
description: Separating information quality from reproductive consequences in evolutionary experiments.
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

# Cue Reliability and Offspring Interventions

## Overview
A cue intervention changes information available before a choice. An offspring intervention changes reproductive consequences. Separate these interventions to test whether a fitted observation model predicts behavior and whether an action-value model predicts reproductive success.

## Experiment-design reference card
- Declare environment e, cue c, action or expressed phenotype a, and descendant count at a specified inclusive census.
- Cue arm: vary P(c | e) while holding environmental frequencies, resources, action costs and offspring consequences fixed as far as the design permits.
- Consequence arm: change offspring consequences conditional on environment and action while holding cue statistics fixed. Freeze the controller during a short-term test; study adaptation in a separately identified arm.
- Factorial arm: cross cue reliability with consequence changes; record interactions rather than attributing every behavioral shift to sensing.
- Controls: cue-shuffled and cue-blind comparators, matched intervention costs, held-out choices, and independently measured offspring counts including zeros.
- Temporal controls: match environmental frequencies while varying their ordering; distinguish private cues from a shared environmental history.

## Growth and inference
For independent environmental states with probabilities p(e), allocation pi(a), and nonnegative growth factors f(a,e), a simple large-population benchmark is sum_e p(e) log(sum_a pi(a) f(a,e)), where each inner sum is positive. A finite-horizon count, survival-conditioned endpoint, and long-run log-growth rate are different outcomes. State which is measured.

In the special horse-race case, an action reproduces only in its matching environment and unconstrained allocation maximizes expected log growth by matching environmental probabilities, or their cue-conditioned probabilities when a cue is available. This model-specific result does not identify expected free energy with offspring count. Serial dependence, inherited state, density regulation and finite-population extinction need additional models.

## Relation to Active Inference
Evaluate the observation mapping with cue interventions and evaluate the proposed link between action value and reproductive value with consequence interventions. Success on one test does not establish the other link.

## Related concepts
- [[phenotypic_plasticity]]
- [[natural_selection]]
- [[cohort_census_design]]
- [[multilag_stratified_preparations]]

## References
- Kelly, J. L., Jr. (1956). A new interpretation of information rate. Bell System Technical Journal, 35, 917–926. https://doi.org/10.1002/j.1538-7305.1956.tb03809.x

The factorial protocol is a proposed diagnostic design; no empirical effect size or result is asserted.
