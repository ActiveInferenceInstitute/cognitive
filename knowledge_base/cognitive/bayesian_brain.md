---
title: Bayesian Brain
type: concept
status: stable
tags:
- cognition
- computation
- probability
- inference
- neuroscience
semantic_relations:
  - type: implements
    links:
      - "[[predictive_processing]]"
  - type: related
    links:
      - "[[free_energy_principle]]"
      - "[[active_inference]]"
      - "[[perceptual_inference]]"
---


# Bayesian Brain


## Overview

The Bayesian Brain hypothesis proposes that the brain implements Bayesian inference to process sensory information, make decisions, and generate actions. This framework suggests that neural computations can be understood as probabilistic inference operations that combine prior knowledge with incoming sensory evidence to form optimal posterior beliefs about the world.

## Core Principles

### Probabilistic Inference

- [[bayesian_inference]] - Core computational principle

  - [[prior_beliefs]] - Existing knowledge

    - [[learned_priors]] - Experience-based

    - [[structural_priors]] - Architecture-based

  - [[likelihood_computation]] - Computing how well hypotheses explain the data

    - [[sensory_likelihood]] - Probability of sensory input given a cause

    - [[causal_likelihood]] - Cause-effect relations

### Uncertainty Processing

- [[uncertainty_representation]] - How the brain encodes probability distributions

  - [[sensory_uncertainty]] - Input noise

  - [[model_uncertainty]] - Uncertainty about which model is correct

  - [[structural_uncertainty]] - Uncertainty about model structure

### Hierarchical Organization

- [[hierarchical_inference]] - Multi-level processing

  - [[hierarchical_priors]] - Level-specific knowledge

  - [[hierarchical_likelihood]] - Level-specific evidence

  - [[posterior_propagation]] - Between-level updates

## Neural Implementation

### Circuit Mechanisms

- [[probabilistic_circuits]] - Neural circuits computing with probabilities

  - [[population_coding]] - Distributed representation

    - [[probabilistic_population]] - Population activity encoding probability distributions

    - [[neural_sampling]] - Neural activity as samples from a distribution

  - [[message_passing]] - Exchange of beliefs between neural units

    - [[belief_propagation]] - Passing messages to update marginal beliefs

    - [[error_propagation]] - Passing prediction errors through the hierarchy

### Synaptic Mechanisms

- [[synaptic_computation]] - Local processing

  - [[synaptic_weights]] - Connection strengths encoding learned priors

  - [[synaptic_plasticity]] - Activity-dependent changes in synaptic strength

  - [[dendritic_computation]] - Input integration within dendrites

### Network Architecture

- [[bayesian_networks]] - Directed graphical models of dependencies

  - [[directed_connections]] - Causal structure

  - [[reciprocal_connections]] - Two-way links between neural areas

  - [[lateral_connections]] - Within-level interaction

## Computational Framework

### Mathematical Formulation

```math

P(h|d) = \frac{P(d|h)P(h)}{P(d)}

```

where:

- $P(h|d)$ is the posterior probability

- $P(d|h)$ is the likelihood

- $P(h)$ is the prior probability

- $P(d)$ is the evidence

### Inference Algorithms

- [[variational_inference]] - Approximate posterior inference by optimization

  - [[mean_field]] - Independence assumption

  - [[laplace_approximation]] - Gaussian approximation around the posterior mode

  - [[particle_filtering]] - Sequential Monte Carlo state estimation

### Learning Mechanisms

- [[bayesian_learning]] - Updating beliefs from experience

  - [[parameter_learning]] - Estimating model parameters from data

  - [[structure_learning]] - Model discovery

  - [[meta_learning]] - Learning to learn

## Cognitive Functions

### Perception

- [[bayesian_perception]] - Perception as probabilistic inference about causes

  - [[cue_integration]] - Multi-sensory fusion

  - [[perceptual_inference]] - Inferring hidden causes of sensations

  - [[perceptual_learning]] - Improved perception through experience

### Decision Making

- [[bayesian_decision]] - Choosing actions by expected utility

  - [[value_computation]] - Utility estimation

  - [[risk_assessment]] - Evaluating outcome probabilities and costs

  - [[policy_selection]] - Action choice

### Learning and Memory

- [[bayesian_learning]] - Updating beliefs from experience

  - [[model_updating]] - Belief revision

  - [[structure_learning]] - Learning model structure from data

  - [[memory_formation]] - Encoding experiences into memory

## Applications

### Clinical Applications

- [[psychiatric_disorders]] - Conditions framed as disturbances of inference

  - [[schizophrenia]] - Linked to aberrant inference and prediction error

  - [[autism]] - Linked to atypical weighting of priors

  - [[anxiety]] - Linked to heightened response to uncertainty

### Artificial Intelligence

- [[bayesian_ai]] - Bayesian methods in artificial intelligence

  - [[probabilistic_programming]] - Languages for specifying probabilistic models

  - [[bayesian_networks]] - Directed graphical models of dependencies

  - [[bayesian_deep_learning]] - Neural networks with uncertainty estimates

### Cognitive Technology

- [[brain_machine_interfaces]] - Devices linking neural activity to machines

  - [[decoder_design]] - Decoding intent from neural signals

  - [[feedback_optimization]] - Tuning interface feedback to improve control

  - [[adaptive_interfaces]] - Interfaces that adapt to the user

## Research Methods

### Experimental Paradigms

- [[psychophysics]] - Relating stimuli to perceptual judgments

  - [[threshold_estimation]] - Sensitivity measurement

  - [[uncertainty_measurement]] - Measuring confidence in judgments

  - [[prior_manipulation]] - Altering expectations to test priors

### Neural Recording

- [[neuroimaging]] - Brain measurement

  - [[fmri_studies]] - Spatial activity patterns from fMRI

  - [[eeg_recording]] - Millisecond-scale electrical activity

  - [[single_unit_recording]] - Recording activity of individual neurons

### Computational Modeling

- [[bayesian_models]] - Models for testing Bayesian theories of cognition

  - [[generative_models]] - Models of how observations arise from causes

  - [[inference_models]] - Models of how beliefs are computed

  - [[learning_models]] - Models of how beliefs and parameters change

## Theoretical Extensions

### Information Theory

- [[information_processing]] - Transformation of signals in neural systems

  - [[entropy_computation]] - Uncertainty measure

  - [[mutual_information]] - Dependency measure

  - [[free_energy]] - Bound on surprise minimized by inference

### Complexity Theory

- [[computational_complexity]] - Processing demands

  - [[algorithmic_complexity]] - Resource cost of an algorithm

  - [[sample_complexity]] - Data requirements

  - [[model_complexity]] - Number and flexibility of model parameters

### Emergence Theory

- [[emergent_properties]] - Properties arising from component interactions

  - [[collective_computation]] - Computation arising from many interacting units

  - [[self_organization]] - Spontaneous order

  - [[criticality]] - Operating near a phase transition

## Future Directions

### Current Challenges

- scalability - Large-scale inference

- [[biological_plausibility]] - Whether algorithms can run in real neurons

- [[real_time_processing]] - Speed requirements

### Emerging Applications

- [[neuroprosthetics]] - Devices restoring neural function

- cognitive enhancement - Performance improvement

- [[artificial_consciousness]] - Machine awareness

## References

- [[knill_pouget_bayesian]]

- [[friston_free_energy]]

- [[doya_bayesian_brain]]

- [[ma_probabilistic_population]]

## Related Concepts

- [[predictive_processing]]

- [[active_inference]]

- [[free_energy_principle]]

- [[probabilistic_computation]]

- [[neural_coding]]

- [[cognitive_architecture]]

