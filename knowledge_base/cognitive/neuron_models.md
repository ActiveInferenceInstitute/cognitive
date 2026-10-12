---
title: Neuron Models
description: Mathematical descriptions of neuronal activity, from conductance-based Hodgkin-Huxley equations to integrate-and-fire and rate models, and the trade-off between biological detail and computational cost.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - computational_neuroscience
  - spiking_neurons
  - biophysics
semantic_relations:
  - type: relates
    links:
      - "[[ion_channels]]"
      - "[[firing_rate]]"
      - "[[spike_trains]]"
      - "[[dendritic_computation]]"
      - "[[neural_computation]]"
---

# Neuron Models

## Overview
A neuron model is a mathematical description of how a nerve cell converts inputs into electrical activity. Models differ in the level of detail they keep. Some track the flow of ions through membrane channels, some reduce the cell to a single voltage variable with a spiking rule, and some summarise output as a continuous firing rate. The choice depends on the question, since extra biophysical detail brings more parameters and computational cost.

## Main model classes
- **Hodgkin-Huxley.** A conductance-based model of the squid giant axon in which voltage-dependent sodium and potassium conductances, with gating variables, produce the action potential (Hodgkin and Huxley, 1952; see [[ion_channels]]). Many later conductance-based models extend it with other channels.
- **Leaky integrate-and-fire.** The membrane potential obeys $\tau_m \dot V = -(V - V_{\mathrm{rest}}) + R\,I(t)$, and when $V$ reaches a threshold a spike is recorded and $V$ is reset. It is simple and widely used in network studies (Gerstner and colleagues, 2014).
- **Reduced two-variable models.** The FitzHugh-Nagumo and Izhikevich models capture bursting, adaptation and other firing patterns with few parameters and efficient simulation (Izhikevich, 2003).
- **Rate models.** Unit activity is a smooth function of summed input, as in artificial neural networks and many models of cortical dynamics (see [[firing_rate]]).
- **Multicompartment models.** Dendrites and axons are divided into coupled compartments using cable theory, allowing study of dendritic integration (see [[dendritic_computation]]).

## Choosing a model
Integrate-and-fire models can reproduce spike timing of cortical neurons well under some input conditions, but they do not capture ion channel dynamics or detailed dendritic processing. Detailed models allow study of drugs and channel mutations but demand more data than are usually available. Network models of cognition therefore generally use reduced units (see [[neural_computation]] and [[spike_trains]]).

## Relation to Active Inference
Active Inference models of neurons are mostly at a higher level of abstraction: neuronal activity is treated as the gradient descent of variational free energy, and populations as encoding expectations and prediction errors. In dynamic causal modelling, neural mass models describe the average activity of populations and are fitted to electrophysiological data. These formulations do not replace spiking or conductance-based models and operate at a different scale.

## Related concepts
- [[ion_channels]] — the molecular basis of conductance models
- [[firing_rate]] — the variable in rate-based models
- [[spike_trains]] — the output that spiking models try to reproduce
- [[dendritic_computation]] — the target of multicompartment models
- [[excitability]] — the property that models aim to capture

## References
- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. Journal of Physiology, 117, 500-544.
- Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks, 14, 1569-1572.
- Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. Cambridge University Press.
