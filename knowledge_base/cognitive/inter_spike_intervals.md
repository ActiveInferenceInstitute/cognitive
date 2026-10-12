---
title: Inter-Spike Intervals
description: The time between successive action potentials of a neuron, their distribution and variability measures, evidence of irregular cortical firing, and implications for neural coding.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neural_coding
  - spike_statistics
  - neuroscience
semantic_relations:
  - type: relates
    links:
      - "[[spike_trains]]"
      - "[[firing_rate]]"
      - "[[spike_timing]]"
      - "[[neural_coding]]"
      - "[[neuron_models]]"
---

# Inter-Spike Intervals

## Overview
An inter-spike interval (ISI) is the time between two consecutive action potentials of the same neuron. A spike train is described fully by the sequence of its spike times, and the sequence of ISIs describes the same information in a different form. The distribution of ISIs is a basic summary of firing statistics and a diagnostic of the underlying neural mechanisms, such as refractoriness, bursting and adaptation.

## Distributions and measures
For a homogeneous Poisson process with rate $r$, the ISIs are independent and exponentially distributed, $p(\tau)=r\,e^{-r\tau}$, with mean $1/r$ and standard deviation $1/r$. The coefficient of variation, $C_V=\sigma_{\mathrm{ISI}}/\mu_{\mathrm{ISI}}$, is therefore 1 for a Poisson process, near 0 for a perfectly regular neuron, and above 1 for bursty firing. Real neurons deviate from the Poisson description at short intervals, because the absolute refractory period of roughly a millisecond and the relative refractory period suppress very short ISIs. Related measures are the Fano factor of the spike count, and the serial correlation between successive intervals. Bursting neurons often show a bimodal ISI histogram, with short intra-burst and long inter-burst intervals (see [[spike_trains]]).

## Irregularity in cortex
Cortical neurons recorded in awake animals typically have a $C_V$ near 1, so their firing is highly irregular. Softky and Koch (1993) argued that this is hard to reconcile with a neuron that integrates many excitatory inputs over a time window, since averaging should regularise output, a point that became known as a paradox. Proposed resolutions include a balance of excitation and inhibition that keeps the membrane potential near threshold so that fluctuations trigger spikes. Shadlen and Newsome (1998) analysed the variability of cortical discharge and its implications for rate and temporal codes. Irregularity is consistent with a rate code with noise, and also does not exclude information in precise timing (see [[firing_rate]] and [[first_spike_latency]]).

## Relation to Active Inference
The link is weak. Active Inference process theories use firing rates as expectations, and stochastic spiking may be treated as sampling from the underlying rate, so ISI statistics matter for how a population can represent uncertainty. Specific predictions of the framework for ISI distributions are not established.

## Related concepts
- [[spike_trains]] — the sequences described by their intervals
- [[firing_rate]] — the mean quantity set by the average interval
- [[spike_timing]] — information carried by exact times
- [[neural_coding]] — how spikes represent information
- [[neuron_models]] — models that generate ISI statistics

## References
- Softky, W. R., & Koch, C. (1993). The highly irregular firing of cortical cells is inconsistent with temporal integration of random EPSPs. Journal of Neuroscience, 13, 334-350.
- Shadlen, M. N., & Newsome, W. T. (1998). The variable discharge of cortical neurons: Implications for connectivity, computation, and information coding. Journal of Neuroscience, 18, 3870-3896.
- Dayan, P., & Abbott, L. F. (2001). Theoretical Neuroscience. Cambridge, MA: MIT Press.
