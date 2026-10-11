---
title: Reproducible Research
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - methodology
  - reproducibility
  - research_practice
semantic_relations:
  - type: relates
    links:
      - "[[operationalism]]"
      - "[[scientific_method]]"
      - "[[active_inference]]"
aliases:
  - reproducibility
---

# Reproducible Research

## Overview
Reproducible research is the practice of publishing the data, code and specification of a computational analysis so that others can regenerate the reported results. The term traces to Claerbout's work in geophysics and was argued for in computational science by Peng (2011), who proposed it as a minimum standard when full independent replication is impractical.

## Core claims
**Terminology.** The National Academies report (2019) defines reproducibility as obtaining consistent results using the same input data, computational steps, methods and code, and replicability as obtaining consistent results across studies that collect new data. Other fields use the two words the other way round, so any statement should define its terms.

**What must be specified.**
- Code and its version, and the environment and dependency versions that run it.
- Input data, or a deterministic procedure for generating it.
- Every source of randomness, with seeds recorded.
- Parameters and configuration, kept as files rather than command-line memory.
- The mapping from raw outputs to each reported number or figure.

**Why it fails in practice.** Unrecorded seeds, silent dependency drift, manual spreadsheet steps, and results pasted into prose without provenance. Computational reproducibility is necessary but not sufficient for a correct result: a perfectly reproducible analysis can still be wrong.

**Link to operationalism.** A complete executable analysis is an operational definition of its results: the quantity is whatever the specified procedure computes (see [[operationalism]]).

## Role in Active Inference
Active-inference simulations are stochastic whenever they use sampling-based belief updates, stochastic action selection, or random initial conditions, so seeding matters. In this repository `InferenceConfig` carries a `seed` (default 0) that initializes the dispatcher's random generator, which makes sampling-based inference repeatable. Agents are configured through YAML files with explicit matrix shapes (see the `A`, `B`, `C`, `D`, `E` conventions in `DiscreteGenerativeModel`), which makes a model run a recorded specification rather than a script edit.

Reproducibility also covers changes in the method itself. When the expected free energy objective was corrected to the canonical risk-plus-ambiguity form, the changelog documented the behaviour change and kept the old objective reachable through `epistemic_weight=1.0` (or `exploration_weight=1.0` on `InferenceConfig`), so earlier results can be regenerated. Tests that pin mathematical identities guard against such drift going unnoticed.

## Related concepts
- [[operationalism]] — executable analysis as operational definition
- [[scientific_method]] — replication as an epistemic norm
- [[experiment_design]] — planning analyses before running them
- [[validation_methods]] — checking that a pipeline computes what is claimed
- [[model_comparison]] — comparisons are only meaningful when runs are reproducible
- [[monte_carlo_methods]] — stochastic procedures that need seed control

## References
- Peng, R. D. (2011). Reproducible Research in Computational Science. *Science*.
- National Academies of Sciences, Engineering, and Medicine (2019). *Reproducibility and Replicability in Science*.
