---
title: Quality Metrics
type: concept
status: stable
updated: 2026-10-09
tags:
- metrics
- evaluation
- calibration
semantic_relations:
  - type: relates
    links:
      - "[[validation]]"
      - "[[information_theory]]"
      - "[[expected_free_energy]]"
      - "[[knowledge_base/cognitive/performance_metrics]]"
      - "[[knowledge_base/cognitive/quality_assessment]]"
---


# Quality Metrics

Common metrics for cognitive models and inferences:

- Log likelihood / ELBO improvements

- Calibration (ECE), Brier score, NLL

- Posterior entropy and information gain

- Free-energy based scores (expected free energy components)

See: [[validation]], [[information_theory]], [[expected_free_energy]].

## What Each Family Measures

**Fit.** Log likelihood and the evidence lower bound (ELBO) say how well a model explains observed data. In Active Inference the negative ELBO is the variational free energy, so a model that lowers it explains the data better, subject to a complexity penalty from the divergence between posterior and prior. Fit alone rewards overfitting, which is why model comparison uses free energy or cross-validation instead of raw likelihood ([[knowledge_base/cognitive/model_selection]]).

**Probabilistic accuracy and calibration.** Negative log likelihood (NLL) and the Brier score are proper scoring rules: they are minimised in expectation by reporting true probabilities. Expected calibration error (ECE) bins predictions by confidence and averages the gap between confidence and observed frequency. It is a diagnostic and not a proper scoring rule, and its value depends on the binning.

**Uncertainty and information.** Posterior entropy shows how much uncertainty remains after inference; the drop in entropy from prior to posterior is the information gained from an observation ([[knowledge_base/mathematics/information_theory]]).

**Decision quality.** Expected free energy decomposes into risk (divergence of predicted from preferred outcomes) and ambiguity (expected observation entropy), plus an epistemic term that can be read as expected information gain ([[knowledge_base/mathematics/expected_free_energy]]). Tracking the components separately shows whether an agent is failing at goal pursuit or at uncertainty reduction.

**Cost.** Runtime and memory are quality metrics for practical use ([[knowledge_base/cognitive/performance_metrics]]).

## Caveat on Free-Energy Scores

Free energy is not comparable across models with different observation spaces, because its scale depends on what is predicted. Use it to compare variants on the same data, and report an external task metric alongside it.

## What the Repository Computes

Implemented quantities, from `code/tools/src/models/active_inference` and `code/Things`:

- `ModelState` carries `free_energy` and `prediction_error`. In `ActiveInferenceModel.calculate_free_energy` the value is half the prediction error plus the KL divergence of beliefs from the prior, a compact proxy and not a full variational bound.
- `DiscreteGenerativeModel.expected_free_energy` returns total, risk, ambiguity and an epistemic diagnostic. The epistemic term is not added to the total a second time, a point pinned by a test.
- `SimplePOMDP` records per-step `free_energy`, `efe_total`, `efe_epistemic` and `efe_pragmatic` histories.
- `MatrixOps.compute_entropy` and `compute_kl_divergence` supply entropy and divergence.
- `ContinuousActiveInference` reports a precision-weighted sensory and dynamics free energy.
- `cognitive-benchmark` reports `seconds_per_call` for matrix operations, dispatcher inference, Simple POMDP and continuous inference.

The codebase has no function for ECE, the Brier score or held-out likelihood. Documentation quality (link integrity, frontmatter) is checked by separate scripts in `code/scripts`, which concern the vault rather than model quality; see [[knowledge_base/cognitive/quality_assessment]].
