---
title: Environmental Monitoring
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - systems
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/systems/environmental_applications]]"
      - "[[knowledge_base/systems/internet_of_things]]"
      - "[[knowledge_base/systems/digital_twins]]"
      - "[[knowledge_base/BioFirm/ecological_active_inference]]"
---

# Environmental Monitoring

## Overview

Environmental monitoring is the repeated measurement of air, water, soil, climate and ecological indicators to estimate the state of an ecosystem and detect change. Measurement is limited by sensor cost, power, access and the time it takes to analyse data, so choosing what to measure, where and when is as important as the estimator itself. Active Inference offers a single formulation for these questions: a model of the environment predicts observations, surprising observations flag anomalies, and the value of the next measurement is its expected reduction in uncertainty. The state-estimation and sensor-selection parts rest on long-established Bayesian methods; presenting them as Active Inference is a reframing, and applications to real monitoring networks are proposals in this repository.

## Main Methods

**State estimation.** Kalman and particle filters, and data assimilation in weather and hydrology, combine a dynamical model with noisy observations to estimate hidden states. This is the same computation as belief updating under a generative model ([[knowledge_base/cognitive/belief_updating]]).

**Adaptive sampling.** Instead of a fixed schedule, the network samples where a model is most uncertain. Krause, Singh and Guestrin (2008) showed how to place sensors in a Gaussian process model by maximising mutual information, with a near-optimal greedy guarantee. Expected information gain used in Active Inference is the same quantity ([[knowledge_base/cognitive/epistemic_foraging]]), though Active Inference additionally trades it against preferences about outcomes, for example keeping a river temperature within a range.

**Anomaly detection.** A model trained on normal behaviour predicts the next reading, and a large precision-weighted prediction error flags an event such as a pollutant release or a failed sensor. Drift, seasonality and sensor faults generate false alarms, so the model must be updated and sensor reliability represented explicitly through precision.

## Trade-offs

An informative sample can cost battery or a site visit, so the objective needs a cost term. A misspecified model makes anomaly scores unreliable. Sparse networks leave the spatial field underdetermined, so uncertainty maps should be reported with estimates.

## In This Repository

Related pages are [[knowledge_base/systems/environmental_applications]], [[knowledge_base/systems/internet_of_things]], [[knowledge_base/biology/ecological_dynamics]], [[knowledge_base/BioFirm/bioregional_state_space]] and [[knowledge_base/BioFirm/ecological_active_inference]]. The BioFirm material specifies a framework for bioregional modelling, and the `code/Things/BioFirm` directory holds documentation including an earth-systems schema, not a monitoring pipeline. The code tree has no sensor ingestion, anomaly detector or adaptive sampler. Its building blocks (belief update, expected free energy with an epistemic component, precision update) are in `code/tools/src/models/active_inference`.

## References

- Krause, A., Singh, A., Guestrin, C. (2008). Near-optimal sensor placements in Gaussian processes: theory, efficient algorithms and empirical studies. *Journal of Machine Learning Research*.

## Related Resources

- [[knowledge_base/systems/environmental_applications]]
- [[knowledge_base/systems/internet_of_things]]
