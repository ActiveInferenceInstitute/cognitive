---
title: Neural Computation
type: concept
status: stable
tags:
- computation
- neuroscience
- information_processing
- brain_function
- modeling
semantic_relations:
  - type: implements
    links: []
  - type: related
    links:
      - "[[knowledge_base/biology/neuroscience]]"
      - "[[information_processing]]"
      - "[[neural_networks]]"
      - "[[cognitive_architecture]]"
---


# Neural Computation


## Overview

Neural Computation describes how the brain processes information through its networks of neurons, implementing cognitive functions through distributed parallel processing. This framework bridges neuroscience and computation, explaining how neural circuits perform information processing and learning.

## Core Principles

### Information Processing

- neural processing - How neural circuits transform information

  - [[signal_processing]] - Filtering and transforming neural signals

    - [[encoding]] - Converting stimuli into neural activity patterns

    - transformation - Mapping input representations onto new ones

  - [[computation_types]] - Categories of neural computation

    - [[local_computation]] - Processing within a small neural circuit

    - [[distributed_computation]] - Processing spread across many regions

### Network Organization

- [[neural_architecture]] - Large-scale organization of neural circuits

  - [[circuit_organization]] - Wiring patterns among neurons

    - [[feedforward]] - One-way flow from input to output

    - [[recurrent]] - Connections that feed activity back into the circuit

  - [[hierarchy_levels]] - Successive tiers of processing

    - processing stages - Sequential steps from input to output

    - [[abstraction_levels]] - Increasingly abstract representations up the hierarchy

### Learning Mechanisms

- [[neural_learning]] - Experience-driven changes in neural circuits

  - [[synaptic_plasticity]] - Activity-dependent change in synaptic strength

    - [[hebbian_learning]] - Synapses strengthen when neurons fire together

    - error driven - Learning that adjusts to reduce prediction error

  - [[network_adaptation]] - Network-wide adjustment to new demands

    - weight updates - Adjusting connection strengths during learning

    - [[architecture_changes]] - Growth or pruning of connections and units

## Computational Elements

### Neural Units

- [[processing_units]] - Neurons as basic computing elements

  - [[neuron_models]] - Mathematical descriptions of neuron behavior

    - [[activation_function]] - Function mapping a unit's input to its output

    - [[threshold_logic]] - Firing only when input exceeds a threshold

  - [[population_coding]] - Information carried by groups of neurons

    - [[rate_coding]] - Information carried by firing rate

    - [[temporal_coding]] - Information carried by spike timing

### Synaptic Operations

- [[connection_compute]] - Computation performed at synapses

  - [[weight_operations]] - Operations that scale signals by synaptic strength

    - multiplication - Scaling one signal by another

    - integration - Summing inputs over space and time

  - modulation - Adjusting the gain or timing of signals

    - gating - Opening or blocking signal transmission

    - [[scaling]] - Adjusting signal magnitude by a factor

### Network Operations

- [[circuit_compute]] - Computation carried out by neural circuits

  - [[pattern_operations]] - Operations on patterns of activity

    - [[pattern_completion]] - Recovering a full pattern from partial input

    - [[pattern_separation]] - Making similar inputs more distinct

  - [[sequence_operations]] - Operations on ordered series of events

    - [[temporal_integration]] - Combining inputs arriving over time

    - [[prediction]] - Anticipating upcoming input

## Implementation Mechanisms

### Circuit Dynamics

- [[neural_dynamics]] - Time-varying behavior of neural activity

  - activity patterns - Spatiotemporal patterns of neural firing

    - [[oscillations]] - Rhythmic, repeating fluctuations in activity

    - [[synchronization]] - Neurons firing in coordinated timing

  - [[state_transitions]] - Shifts between distinct activity states

    - [[attractor_dynamics]] - Activity settling into stable patterns

    - [[trajectory_evolution]] - Path of network state through time

### Information Flow

- [[signal_flow]] - Movement of signals through the network

  - [[feedforward_flow]] - Signals passing from input toward output

    - [[feature_extraction]] - Detecting informative features in the input

    - transformation - Reshaping features into new representations

  - [[feedback_flow]] - Signals passing back to earlier stages

    - [[top_down]] - Higher-level signals modulating lower stages

    - [[prediction_error]] - Mismatch between expected and actual input

### Control Mechanisms

- [[neural_control]] - Mechanisms that regulate neural activity

  - [[gain_control]] - Adjusting response sensitivity

    - amplification - Boosting signal strength

    - suppression - Reducing signal strength

  - gating - Controlling which signals pass through

    - [[selective_routing]] - Directing signals along chosen pathways

    - [[inhibitory_control]] - Using inhibition to block unwanted activity

## Computational Functions

### Representation

- [[neural_coding]] - How neural activity represents information

  - [[feature_representation]] - Neural encoding of stimulus features

    - [[selectivity]] - Preferential response to particular stimuli

    - [[invariance]] - Stable response despite stimulus changes

  - [[distributed_representation]] - Information spread over many units

    - [[population_coding]] - Information carried by groups of neurons

    - [[sparse_coding]] - Representing information with few active units

### Transformation

- [[neural_transform]] - Operations that turn inputs into outputs

  - [[linear_operations]] - Transformations preserving scaling and addition

    - [[projection]] - Mapping activity into a different space

    - [[combination]] - Merging multiple inputs into one output

  - [[nonlinear_operations]] - Transformations that break simple proportionality

    - thresholding - Passing only inputs above a cutoff

    - [[normalization]] - Dividing responses by overall activity level

### Learning

- [[computation_learn]] - Improvement of computations through experience

  - [[supervised_learning]] - Learning from labeled targets or teaching signals

    - [[error_correction]] - Adjusting weights to reduce output error

    - [[gradient_descent]] - Iteratively adjusting weights to reduce error

  - [[unsupervised_learning]] - Learning structure without labeled targets

    - [[hebbian_learning]] - Co-active neurons strengthen their connection

    - [[competitive_learning]] - Units compete and the winner updates its weights

## Applications

### Cognitive Models

- [[brain_models]] - Computational models of cognition

  - [[perception_models]] - Models of sensory perception

    - [[visual_processing]] - Computations in the visual system

    - [[auditory_processing]] - Computations in the auditory system

  - [[memory_models]] - Models of memory formation and recall

    - [[storage]] - Encoding memories in synaptic changes

    - [[retrieval]] - Reactivating stored patterns from partial cues

### Neural Engineering

- [[brain_engineering]] - Engineering that interfaces with or mimics the brain

  - [[neural_prosthetics]] - Devices that substitute for lost neural function

    - [[sensory_devices]] - Prostheses that restore sensory input

    - [[motor_devices]] - Prostheses that restore movement

  - [[brain_interfaces]] - Links between brain activity and devices

    - recording - Reading neural activity into a device

    - stimulation - Driving neural activity with electrical or magnetic input

### Clinical Applications

- [[disorder_models]] - Models of how neural computation fails in disease

  - [[dysfunction]] - Disrupted neural computation

    - [[circuit_problems]] - Faulty wiring or connectivity in circuits

    - dynamic problems - Abnormal activity dynamics, such as seizures

  - intervention - Treatments that aim to correct dysfunction

    - stimulation - Activating neural tissue for therapeutic effect

    - modulation - Altering circuit activity to treat dysfunction

## Research Methods

### Experimental Methods

- [[neural_experiments]] - Experiments probing neural function

  - recording - Measuring neural firing during tasks

    - [[single_unit]] - Recording from individual neurons

    - population - Recording from many neurons at once

  - manipulation - Experimentally altering neural activity

    - stimulation - Activating selected brain regions

    - [[inactivation]] - Temporarily silencing selected brain regions

### Computational Methods

- [[simulation_methods]] - Computer simulation of neural systems

  - [[network_simulation]] - Simulating networks of model neurons

    - [[dynamics]] - Mathematics of how activity evolves over time

    - [[learning]] - Models of how synapses change with experience

  - analysis - Quantitative analysis of model behavior

    - [[stability]] - Whether activity returns to equilibrium after perturbation

    - [[optimization]] - Finding parameters that minimize an objective

### Theory Development

- [[neural_theory]] - Theoretical accounts of neural computation

  - principles - Core organizing rules of neural computation

    - computation - Rules describing what circuits compute

    - organization - Rules describing how circuits are arranged

  - [[predictions]] - Testable predictions derived from the theory

    - [[behavior]] - Behavioral outcomes the theory predicts

    - [[mechanism]] - Neural mechanisms the theory proposes

## Future Directions

### Current Challenges

- [[scale_problems]] - Difficulty scaling models to whole brains

- [[complexity]] - Difficulty of handling many interacting components

- [[biological_realism]] - Fidelity of models to real neural biology

### Emerging Approaches

- new methods - Emerging techniques for studying and modeling circuits

  - [[large_scale]] - Studying activity across large brain areas

    - [[whole_brain]] - Modeling or recording the entire brain

    - [[long_time]] - Tracking neural activity over long periods

  - [[new_theory]] - Emerging theoretical accounts

    - principles - Emerging general principles of computation

    - [[frameworks]] - Emerging conceptual frameworks

## References

- [[dayan_abbott]]

- [[hopfield_networks]]

- [[rumelhart_parallel]]

- [[sejnowski_computational]]

## Related Concepts

- [[information_processing]]

- [[neural_networks]]

- [[cognitive_architecture]]

- brain function

- computation

- [[learning_theory]]

