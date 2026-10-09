---
title: Computational Modeling
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
      - "[[knowledge_base/mathematics/variational_methods]]"
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/cognitive/matrix_specifications]]"
      - "[[knowledge_base/mathematics/pomdp_framework]]"
      - "[[knowledge_base/mathematics/variational_inference]]"
      - "[[knowledge_base/mathematics/message_passing]]"
      - "[[knowledge_base/mathematics/factor_graphs]]"
      - "[[knowledge_base/cognitive/model_selection]]"
      - "[[knowledge_base/mathematics/sensitivity_analysis]]"
      - "[[knowledge_base/mathematics/numerical_stability]]"
      - "[[knowledge_base/cognitive/simulation_studies]]"
---

# Computational Modeling

## Overview

Computational modeling of Active Inference means turning the theory into executable models: specifying a generative model, running perception and action, and fitting or comparing models against data. The same machinery serves three purposes: simulating idealised agents, fitting models to the behaviour of human or animal participants, and building artificial agents.

## Model specification

In the discrete-time formulation, a model is a partially observable Markov decision process defined by a likelihood mapping $A$ from hidden states to observations, transition matrices $B$ for each action, prior preferences $C$ over outcomes, and initial state priors $D$. Policies are sequences of actions, and the repository's standard objects are documented in [[knowledge_base/cognitive/matrix_specifications]] and [[knowledge_base/mathematics/pomdp_framework]]. Continuous-time models use generalised coordinates of motion and differential equations. Hierarchical models stack several such levels.

## Inference and optimisation

Beliefs about states are updated to reduce variational free energy. Implementations use several schemes: gradient descent on free energy, fixed-point iteration, and message passing on a factor graph (belief propagation and variational message passing, with the Bethe approximation as a related objective). Policy selection evaluates expected free energy for each candidate policy, which scales exponentially with horizon and so motivates tree search, sampling and amortised policy networks. Numerical care is needed, notably in the stability of log-probabilities and softmax functions, and in the choice of the precision and temperature parameters.

## Software ecosystem

Established implementations include the DEM and spm_MDP routines in the SPM neuroimaging package for MATLAB, pymdp for Python (Heins and colleagues, 2022) for discrete state spaces, and RxInfer.jl for message-passing inference in Julia. Choices among them involve trade-offs between fidelity to the published formulation, speed and extensibility. This repository includes its own Python implementation and notes on RxInfer integration under `docs/implementation/rxinfer`.

## Model comparison and validation

Fitting to behaviour proceeds by maximising the likelihood of observed choices under model parameters, usually by variational Bayes or Markov chain Monte Carlo. Competing models are compared by approximate model evidence, with penalties for complexity, and by cross-validation. Good practice includes parameter-recovery tests on synthetic data (checking that known parameters can be recovered from simulated behaviour), sensitivity analysis, and model-recovery checks. A frequent failure is that parameters in these models trade off against each other, so that point estimates are poorly identified even when a model fits well.

## Status

These practices are established methods, borrowed largely from Bayesian statistics and computational neuroscience. The current limitations are scalability, with exact computation over policies feasible only for small problems, and the unresolved matter of how to choose a generative model structure, which is typically fixed by the modeller.

## In this repository

See [[knowledge_base/mathematics/variational_inference]], [[knowledge_base/mathematics/message_passing]], [[knowledge_base/mathematics/factor_graphs]], [[knowledge_base/cognitive/model_selection]], [[knowledge_base/mathematics/sensitivity_analysis]], [[knowledge_base/mathematics/numerical_stability]] and [[knowledge_base/cognitive/simulation_studies]].

## References

- Heins, C., et al. (2022). pymdp: a Python library for active inference in discrete state spaces. *Journal of Open Source Software*, 7(73), 4098.
- Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2020). Active inference on discrete state-spaces: a synthesis. *Journal of Mathematical Psychology*, 99, 102447.

## Related Resources

- [[knowledge_base/mathematics/variational_methods]]
- [[knowledge_base/cognitive/active_inference]]
