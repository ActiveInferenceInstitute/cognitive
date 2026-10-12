---
title: Spike Trains
description: The sequences of action potentials fired by neurons over time, their statistical description by firing rate, interspike intervals and point-process models, and their use in neural coding.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neural_coding
  - spiking_neurons
  - statistics
semantic_relations:
  - type: relates
    links:
      - "[[spike_timing]]"
      - "[[inter_spike_intervals]]"
      - "[[neural_coding]]"
      - "[[population_coding]]"
      - "[[neuron_models]]"
---

# Spike Trains

## Overview
A spike train is the record of the action potentials fired by a neuron, or a set of neurons, over time. Because action potentials are brief and stereotyped, a spike train is usually treated as a series of event times $t_1, t_2, \dots$ and written as a sum of delta functions, $\rho(t)=\sum_i \delta(t-t_i)$. It is the basic data of extracellular recording, and the object that theories of neural coding try to interpret.

## Description and analysis
- **Firing rate.** The count of spikes in a time window, divided by the window length. It can be averaged over time, over repeated trials (the peristimulus time histogram), or over a population.
- **Interspike intervals.** The distribution of times between successive spikes. It shows refractoriness, bursting and regularity (see [[inter_spike_intervals]]).
- **Variability.** The Fano factor, the variance of spike counts divided by their mean, and the coefficient of variation of intervals both equal 1 for a homogeneous Poisson process. Many cortical neurons fire irregularly, with values near 1 (Softky & Koch, 1993), whereas the absolute refractory period makes real neurons more regular at short intervals than a Poisson process.
- **Correlations.** Cross-correlograms and noise correlations between neurons measure shared variability and synchrony.
- **Raster plots.** Display of spike times across trials or neurons.

Statistical models treat spikes as a point process. The inhomogeneous Poisson process, with a time-varying rate $\lambda(t)$, is the simplest, and generalised linear models add dependence on stimulus, spike history and coupling between neurons. Dayan and Abbott (2001) give a standard treatment. Decoding applies Bayesian or regression methods to estimate the stimulus from spike trains.

## Interpretation
Whether information resides in rates or in precise timing is the central interpretive question (see [[spike_timing]]). Variability from trial to trial may be noise, or may reflect internal state not controlled by the experimenter, and so the Poisson description is a convenient approximation, not a claim about mechanism.

## Relation to Active Inference
Active Inference models usually do not simulate spikes, treating neural states as continuous quantities whose dynamics approximate gradient descent on free energy. Probabilistic accounts of cortex that interpret spike variability as sampling from a posterior, related to [[neural_sampling]], connect spike statistics to inference, although whether neurons actually sample is unsettled.

## Related concepts
- [[spike_timing]] — the use of exact spike times
- [[inter_spike_intervals]] — the distribution of gaps between spikes
- [[neural_decoding]] — reading out stimuli from spikes
- [[population_coding]] — information in many spike trains together
- [[neural_sampling]] — spike variability as sampling

## References
- Softky, W. R., & Koch, C. (1993). The highly irregular firing of cortical cells is inconsistent with temporal integration of random EPSPs. Journal of Neuroscience, 13, 334-350.
- Rieke, F., Warland, D., de Ruyter van Steveninck, R., & Bialek, W. (1997). Spikes: Exploring the Neural Code. MIT Press.
- Dayan, P., & Abbott, L. F. (2001). Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems. MIT Press.
