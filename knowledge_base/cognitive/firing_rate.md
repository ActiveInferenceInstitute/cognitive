---
title: Firing Rate
description: The number of action potentials a neuron emits per unit time, how it is estimated, its role as the classical neural code, and its interpretation in process theories of Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neural_coding
  - spikes
  - neuroscience
semantic_relations:
  - type: relates
    links:
      - "[[neural_coding]]"
      - "[[spike_trains]]"
      - "[[inter_spike_intervals]]"
      - "[[population_coding]]"
      - "[[neural_tuning]]"
---

# Firing Rate

## Overview
The firing rate of a neuron is the number of action potentials it generates per unit time, usually expressed in spikes per second (hertz). It is the oldest and most widely used description of neural activity. Adrian (1926) recorded from sensory nerve fibres and found that stronger stimuli produced more impulses per second while the size of each impulse stayed the same, showing that intensity was carried by spike frequency and not spike amplitude.

## Definitions and estimation
Because a spike train is a sequence of discrete events, "rate" needs a precise definition (Dayan and Abbott, 2001). Common versions are:
- **Spike-count rate**: the number of spikes in a trial window divided by its length. This is simple but ignores variation within the window.
- **Peristimulus time histogram**: spike counts in short time bins averaged across repeated trials, giving a time-dependent rate.
- **Population rate**: spikes across many similar neurons in a short interval, which can follow fast changes.
- **Kernel estimates**: smoothing of spike times to give a continuous rate.

Cortical spike counts are often close to Poisson in variability, so that the variance is roughly proportional to the mean, although the fit is imperfect (see [[inter_spike_intervals]]).

## Rate coding and its limits
A rate code assumes that the information is carried by the average spike count, with exact timing carrying little. This works well for tuning curves, such as a neuron's firing rate as a function of stimulus orientation or movement direction (see [[neural_tuning]]). However, behaviour can be fast, and some neurons carry information in precise timing or latency (see [[first_spike_latency]] and [[spike_timing]]). Rieke and colleagues (1997) showed in several systems that spike timing can carry additional information beyond what the rate gives. Rate and timing codes are therefore treated as complementary, and which dominates depends on the system and the task.

## Relation to Active Inference
In process theories of Active Inference, neuronal activity is cast as encoding the expected values of hidden states, and firing rate is related to these expectations through a sigmoid or softmax-like mapping, while prediction errors drive their updates (Friston et al., 2017). This is a proposal about how the variational updates could be implemented by neural dynamics. It is a theoretical mapping, and its quantitative match to recorded firing rates is limited.

## Related concepts
- [[neural_coding]] — the general question of how neurons carry information
- [[spike_trains]] — the event sequences from which rates are computed
- [[inter_spike_intervals]] — timing statistics complementing rate
- [[population_coding]] — information spread across many neurons
- [[neural_tuning]] — the dependence of rate on stimulus

## References
- Adrian, E. D. (1926). The impulses produced by sensory nerve endings: Part I. Journal of Physiology, 61, 49-72.
- Rieke, F., Warland, D., de Ruyter van Steveninck, R., & Bialek, W. (1997). Spikes: Exploring the Neural Code. Cambridge, MA: MIT Press.
- Dayan, P., & Abbott, L. F. (2001). Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems. Cambridge, MA: MIT Press.
- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: A process theory. Neural Computation, 29, 1-49.
