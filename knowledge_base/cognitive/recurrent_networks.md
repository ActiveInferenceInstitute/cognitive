---
title: Recurrent Networks
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - recurrent_networks
  - neural_networks
  - sequence_modeling
  - dynamics
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/neural_architectures]]"
      - "[[knowledge_base/cognitive/deep_learning]]"
      - "[[knowledge_base/cognitive/temporal_models]]"
      - "[[knowledge_base/cognitive/working_memory]]"
---

# Recurrent Networks

## Overview
A recurrent neural network (RNN) is a network with feedback connections, so that its units carry an internal state that depends on past inputs. This makes recurrent networks suited to sequences and to modeling dynamical systems, and they are also a standard abstraction of recurrent connectivity in cortex.

## Mechanism
In the basic form, the hidden state updates as h_t = f(W_h h_(t-1) + W_x x_t + b), with output y_t = g(W_y h_t). Elman (1990) used such networks to study temporal structure in language. Training uses backpropagation through time, which unrolls the network across time steps; gradients are then products of many Jacobians and tend to vanish or explode over long spans (Hochreiter 1991; Bengio, Simard and Frasconi 1994), so basic RNNs struggle with long-range dependencies. Long short-term memory (Hochreiter and Schmidhuber 1997) introduced gated memory cells with additive updates that preserve gradients, and the gated recurrent unit (Cho and colleagues 2014) is a lighter variant. Reservoir computing (echo state networks, Jaeger 2001; liquid state machines, Maass and colleagues 2002) keeps the recurrent weights fixed and random and trains only a linear readout. Hopfield networks (1982) are recurrent networks with symmetric weights whose dynamics descend an energy function and settle into stored attractor patterns, serving as models of associative memory. Continuous-time RNNs, defined by differential equations, link the architecture to neural population models and to dynamical systems analysis, where fixed points, limit cycles, and line attractors correspond to computations such as memory and integration.

Since 2017, Transformers have replaced RNNs for many language tasks, because attention allows parallel training and direct access to distant positions, though recurrent and state-space models remain competitive where long sequences or constant memory per step are needed.

## Role in Active Inference
Active Inference generative models for time series are also recurrent in a sense: hidden states evolve under a transition model, in the discrete case via the B matrices, and in the continuous case via differential equations in generalized coordinates. The difference is that these are probabilistic models whose states are beliefs updated by inference, whereas an RNN's hidden state is a deterministic function trained by gradient descent. RNNs can serve as flexible transition or observation models inside a larger inference scheme, but the simple identification of an RNN hidden state with a posterior belief is not warranted without a probabilistic interpretation. Cortical recurrent connectivity is the substrate that predictive-coding theories assume for message passing.

## Related concepts
- [[knowledge_base/cognitive/deep_learning]] — the wider family of trainable networks
- [[knowledge_base/cognitive/temporal_models]] — models of time-dependent structure
- [[knowledge_base/cognitive/transition_model]] — the state-dynamics component of an Active Inference generative model
- [[knowledge_base/cognitive/working_memory]] — a function recurrent dynamics are often used to model
- [[knowledge_base/mathematics/neural_dynamics]] — differential-equation treatments of neural populations
- [[knowledge_base/mathematics/generalized_coordinates]] — continuous-time state representation

## References
- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences.
- Elman, J. L. (1990). Finding structure in time. Cognitive Science.
- Hochreiter, S., and Schmidhuber, J. (1997). Long short-term memory. Neural Computation.
