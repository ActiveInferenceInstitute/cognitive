---
title: Activation Function
description: The nonlinear function that maps a unit's weighted input to its output in a neural network, with common choices, their derivatives and how they relate to firing-rate and belief-updating models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neural_networks
  - nonlinearity
  - deep_learning
semantic_relations:
  - type: relates
    links:
      - "[[neural_networks]]"
      - "[[softmax_function]]"
      - "[[neural_computation]]"
      - "[[gradient_descent]]"
      - "[[deep_learning]]"
wikidata: Q4677469
---

# Activation Function

## Overview
An activation function $\varphi$ maps the pre-activation $z=w^\top x+b$ of a unit to its output $a=\varphi(z)$. Its main role is to introduce nonlinearity. Without it, a stack of layers composes linear maps and collapses to a single linear map, however deep the network. With a suitable nonlinearity, even a network with one hidden layer can approximate any continuous function on a compact set arbitrarily well, given enough units.

## Common choices
- **Logistic sigmoid**: $\sigma(z)=1/(1+e^{-z})$, with outputs in $(0,1)$ and derivative $\sigma'(z)=\sigma(z)(1-\sigma(z))$.
- **Hyperbolic tangent**: $\tanh(z)$, with outputs in $(-1,1)$ and zero-centred.
- **Rectified linear unit**: $\max(0,z)$, which is cheap to compute and does not saturate for positive inputs, but has zero gradient for negative ones.
- **Softmax**: acts on a whole layer, $\mathrm{softmax}(z)_i=e^{z_i}/\sum_j e^{z_j}$, producing a probability vector (see [[softmax_function]]).

## Design considerations
Saturating functions such as the sigmoid have small derivatives for large $|z|$, which can make gradients vanish in deep networks trained by gradient descent. The choice also affects initialisation, optimisation speed and whether outputs can be read as probabilities. The universal approximation results require a non-polynomial activation.

## Relation to neuroscience and Active Inference
Activation functions are loosely analogous to the relation between a neuron's input and its firing rate, which is monotonic, thresholded and saturating. In process theories of Active Inference, neuronal firing rates are modelled as sigmoid functions of membrane potential, and belief updating on discrete states is written as a softmax of accumulated log-evidence terms, so that the nonlinearity has an inferential interpretation. In this reading the sigmoid or softmax converts a log-posterior into a probability. These process theories are proposals about neural implementation whose empirical support is still being assessed, and the choice of activation in a machine-learning network need not carry this interpretation.

## Related concepts
- [[neural_networks]] — the models in which units apply activation functions
- [[softmax_function]] — the normalising activation for categorical outputs
- [[neural_computation]] — the broader study of computation in neurons
- [[gradient_descent]] — training requires differentiable activations
- [[neural_coding]] — how firing rates represent information
- [[convolutional_networks]] — architecture that applies nonlinearities after each filter

## References
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems, 2, 303-314.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Friston, K., et al. (2017). Active inference: a process theory. Neural Computation, 29, 1-49.
