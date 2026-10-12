---
title: Synaptic Weights
description: Numerical strengths of connections between neurons, how they change through learning rules, and how they are interpreted as encoding priors and generative model parameters in Bayesian brain theories.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - synaptic_plasticity
  - learning
  - bayesian_brain
semantic_relations:
  - type: relates
    links:
      - "[[synaptic_plasticity]]"
      - "[[hebbian_learning]]"
      - "[[spike_timing_dependent_plasticity]]"
      - "[[bayesian_brain]]"
      - "[[predictive_coding]]"
---

# Synaptic Weights

## Overview
A synaptic weight is a number that summarises how strongly the activity of a presynaptic neuron influences a postsynaptic one. In biological terms it abstracts properties such as the number and size of synaptic contacts, receptor density and the probability and amount of transmitter release. In network models the input to unit $i$ is typically written $\sum_j w_{ij}x_j$, where $x_j$ is the presynaptic activity and $w_{ij}$ the weight, with positive values for excitatory and negative values for inhibitory connections. The set of all weights, the connectivity matrix, determines what a network computes, and changing it is the standard model of learning.

## Learning rules
Weights change through plasticity. The simplest local rule, associated with Hebb (1949), increases a weight when pre- and postsynaptic activity coincide: $\Delta w_{ij}\propto x_j\,y_i$. Pure Hebbian growth is unstable, so models add normalisation, decay or sliding thresholds, and biological counterparts include homeostatic mechanisms (see [[homeostatic_plasticity]]). Spike-timing-dependent plasticity makes the sign and size of change depend on the relative timing of spikes (see [[spike_timing_dependent_plasticity]]). In artificial networks, weights are usually adjusted by gradient descent on a loss, which has no direct biological counterpart in its standard form, although several schemes approximate it with local signals.

## Weights as priors in Bayesian accounts
Bayesian brain theories interpret the brain as a generative model whose parameters must be learned. In hierarchical predictive coding, the weights of backward connections encode the mapping from causes at one level to predictions at the level below, so they play the role of learned parameters of the model, and in this sense of learned priors and likelihood structure (Rao and Ballard, 1999; Friston, 2005). Learning is then a gradient descent on free energy over the weights, which under certain assumptions reduces to a Hebbian-like rule driven by the product of prediction error and presynaptic activity, making it local in space and time.

This interpretation is a theoretical proposal. It specifies what weights should represent and how they should change, but it does not show that cortical synapses implement these rules, and experimental tests are limited by the difficulty of measuring weights in vivo. Many different models, including non-Bayesian ones, can describe the same plasticity data.

## Related concepts
- [[synaptic_plasticity]] — processes that change weights
- [[hebbian_learning]] — the classic local learning rule
- [[spike_timing_dependent_plasticity]] — timing-dependent weight change
- [[bayesian_brain]] — the theoretical frame that reads weights as priors
- [[predictive_coding]] — a scheme in which weights parameterise predictions
- [[neural_plasticity]] — broader changes in neural structure and function

## References
- Hebb, D. O. (1949). The organization of behavior: A neuropsychological theory. Wiley.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: A functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience, 2, 79-87.
- Friston, K. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society B, 360, 815-836.
- Dayan, P., & Abbott, L. F. (2001). Theoretical neuroscience: Computational and mathematical modeling of neural systems. MIT Press.
