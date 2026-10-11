---
title: Hopfield Networks
description: Recurrent networks of binary or continuous units with symmetric weights and an energy function, which act as content-addressable memories by settling into attractor states, with capacity limits and modern variants.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neural_networks
  - associative_memory
  - attractors
semantic_relations:
  - type: relates
    links:
      - "[[attractor_dynamics]]"
      - "[[energy_based_models]]"
      - "[[pattern_completion]]"
      - "[[hebbian_learning]]"
      - "[[neural_computation]]"
---

# Hopfield Networks

## Overview
A Hopfield network is a recurrent network of $N$ units with states $s_i\in\{-1,+1\}$ and symmetric weights $w_{ij}=w_{ji}$ with $w_{ii}=0$. Hopfield (1982) showed that such a network has an energy function
$$
E(\mathbf{s})=-\tfrac12\sum_{i\ne j}w_{ij}s_is_j-\sum_i b_is_i,
$$
and that asynchronous updates, in which one unit at a time takes the sign of its input $\sum_j w_{ij}s_j+b_i$, never increase $E$. Since the energy is bounded, the dynamics converge to a local minimum, a fixed point. The network therefore behaves as a content-addressable memory: a corrupted or partial pattern evolves into the nearest stored pattern.

## Storage and capacity
Patterns $\xi^{\mu}$ are stored with the Hebbian rule $w_{ij}=\frac1N\sum_\mu\xi^{\mu}_i\xi^{\mu}_j$. With random patterns, retrieval is reliable only up to a load of roughly $0.14\,N$ patterns, a result derived with methods from the statistical physics of spin glasses (Amit, Gutfreund & Sompolinsky, 1985). Beyond this limit, spurious mixture states and a collapse of retrieval appear. A continuous-valued version with graded units was introduced in Hopfield (1984). Modern Hopfield networks use sharper energy functions and can store a number of patterns that grows faster than linearly with $N$, and certain forms are mathematically related to the softmax attention used in transformer networks.

## Relation to Active Inference
The link is an analogy of form. Both frameworks describe computation as descent on an energy-like function, and attractor states correspond to perceptual or memory states reached by relaxation. Hopfield networks are not derived from a generative model and have no explicit prediction errors, so they differ from variational schemes in which inference is gradient descent on free energy. Predictive-coding networks have been shown to function as associative memories, and attractor accounts of working memory and perceptual inference draw on the same dynamical idea. Treating a Hopfield network as an Active Inference agent would require specifying a generative model that the network is meant to invert.

## Related concepts
- [[attractor_dynamics]] — the dynamical principle behind retrieval
- [[energy_based_models]] — the broader family including Boltzmann machines
- [[pattern_completion]] — the memory function of the network
- [[hebbian_learning]] — the storage rule
- [[basin_of_attraction]] — the set of states that flow to each stored pattern
- [[neural_computation]] — the field in which the model is used

## References
- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences, 79, 2554-2558.
- Hopfield, J. J. (1984). Neurons with graded response have collective computational properties like those of two-state neurons. Proceedings of the National Academy of Sciences, 81, 3088-3092.
- Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). Storing infinite numbers of patterns in a spin-glass model of neural networks. Physical Review Letters, 55, 1530-1533.
