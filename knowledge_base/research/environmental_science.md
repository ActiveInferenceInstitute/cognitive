---
title: Environmental Science
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - research
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/systems/environmental_applications]]"
      - "[[knowledge_base/systems/ecosystem_management]]"
      - "[[knowledge_base/systems/environmental_monitoring]]"
      - "[[knowledge_base/systems/digital_twins]]"
      - "[[knowledge_base/biology/ecological_dynamics]]"
      - "[[knowledge_base/systems/spatial_ecology]]"
      - "[[knowledge_base/free_energy_principle/biology/ecology]]"
      - "[[knowledge_base/systems/Social-Ecological Systems]]"
wikidata: Q188847
---

# Environmental Science

## Overview

Environmental science studies the physical, chemical and biological processes of the Earth system and how human activity changes them. Its subfields include climate science, ecology, hydrology, atmospheric chemistry and conservation. Methods combine long-term observation (ground stations, satellites, sensor networks), process-based simulation, and statistical inference from incomplete data.

## Main questions and methods

- **Ecosystem modelling.** Population dynamics, food webs, nutrient cycles and responses to disturbance, using differential-equation and agent-based models.
- **Climate and weather.** General circulation models integrate the physics of atmosphere, ocean and land. Forecasting systems combine models with observations through *data assimilation*, methods such as the Kalman filter and variational schemes (3D-Var and 4D-Var) that estimate the state of the system from sparse measurements.
- **Monitoring.** Deciding where and when to sample, how to detect change against natural variability, and how to combine sources.
- **Management under uncertainty.** Adaptive management treats interventions as experiments, updating the model as results arrive.

## Connection to Active Inference

The most direct connection is a formal one. Data assimilation is Bayesian state estimation: a prior from a dynamical model is combined with noisy observations to give a posterior. Perceptual inference in Active Inference is the same type of computation, and the generalised-filtering schemes used in the continuous-time formulation are close relatives of Kalman-type filters. This is an established correspondence of methods, not a new discovery, and it does not mean that Active Inference improves on operational assimilation systems.

The second connection is decision-making. Choosing monitoring effort, or an intervention whose effects are uncertain, is a sequential decision problem under partial observability. Active Inference supplies an objective in which gathering information and achieving outcomes are traded off within one expected free energy. Decision-theoretic formulations of conservation and management are well developed in the existing literature, and the contribution of the free energy formulation to them is, for now, conceptual, with few applications on real environmental data. The framework's use for climate as a whole is largely speculative, and physical climate models are not replaced by it.

Ecosystems and the biosphere have also been described as self-organising systems that maintain themselves in characteristic states. This is a metaphor with some mathematical support at the level of individual organisms and is not a validated account of ecosystems.

## In this repository

See [[knowledge_base/systems/environmental_applications]], [[knowledge_base/systems/ecosystem_management]], [[knowledge_base/systems/environmental_monitoring]], [[knowledge_base/systems/digital_twins]], [[knowledge_base/biology/ecological_dynamics]], [[knowledge_base/systems/spatial_ecology]], [[knowledge_base/free_energy_principle/biology/ecology]], and [[knowledge_base/systems/Social-Ecological Systems]]. A study path is [[active_inference_environmental_applications_path]].

## References

- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82, 35-45.

## Related Resources

- [[knowledge_base/systems/environmental_applications]]
- [[knowledge_base/systems/ecosystem_management]]
