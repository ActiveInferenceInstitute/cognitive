---
title: Deep Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - deep_learning
  - neural_networks
  - machine_learning
  - representation_learning
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/neural_architectures]]"
      - "[[knowledge_base/cognitive/predictive_coding]]"
      - "[[knowledge_base/cognitive/variational_free_energy]]"
      - "[[knowledge_base/free_energy_principle/implementations/neural_networks]]"
aliases:
  - deep_neural_networks
---

# Deep Learning

## Overview
Deep learning is the branch of machine learning that uses artificial neural networks with many layers to learn hierarchical representations directly from data. It is trained by gradient descent on a loss function, with gradients computed by backpropagation (Rumelhart, Hinton and Williams 1986), and became dominant after large datasets, GPU computing, and architectural advances arrived in the early 2010s.

## Mechanism
A network is a composition of differentiable layers, each applying a linear map and a nonlinearity, so f(x) = f_L(...f_2(f_1(x))). Parameters are adjusted to minimize an empirical loss by stochastic gradient descent or adaptive variants, with the chain rule giving the gradient with respect to every weight in one backward pass. Depth lets the network build features of increasing abstraction, for example edges, parts, and objects in vision (LeCun, Bengio and Hinton 2015). Main architectural families are convolutional networks, which exploit translation structure in images (AlexNet, Krizhevsky and colleagues 2012, was an early large-scale success); recurrent networks and LSTMs for sequences; and Transformers (Vaswani and colleagues 2017), which replace recurrence with self-attention and now underlie large language models. Generative approaches include variational autoencoders (Kingma and Welling 2014), generative adversarial networks, and diffusion models.

Key practical issues are generalization (models with more parameters than data can still generalize, with the reasons only partly understood), regularization, optimization difficulty, data hunger, vulnerability to adversarial examples, and limited interpretability. Standard training yields point estimates of parameters and does not by default quantify uncertainty.

## Role in Active Inference
The two traditions overlap in several specific places. Variational autoencoders maximize an evidence lower bound, which is the negative of variational free energy, so training one is variational inference with an amortized recognition network. Predictive coding networks have been shown to approximate backpropagation under some conditions with local update rules (Whittington and Bogacz 2017), which makes them candidates for biologically plausible learning. Deep neural networks can also parameterize the generative model, policy, or value function of an Active Inference agent, as in the deep Active Inference line of work (for example Millidge 2020). The differences are material: standard deep learning has no built-in epistemic drive, no explicit beliefs over hidden states, and no principled treatment of action as inference. The agents in this repository use explicit matrix-based models (A, B, C, D, E) rather than deep networks, which trades expressiveness for interpretability and exact inference on small problems.

## Related concepts
- [[knowledge_base/cognitive/neural_architectures]] — network designs and their computational properties
- [[knowledge_base/cognitive/recurrent_networks]] — networks with feedback for sequence processing
- [[knowledge_base/cognitive/predictive_coding]] — hierarchical error-minimizing inference in cortex
- [[knowledge_base/cognitive/variational_free_energy]] — the objective shared with variational autoencoders
- [[knowledge_base/cognitive/reinforcement_learning]] — the learning paradigm often combined with deep networks
- [[knowledge_base/cognitive/artificial_intelligence]] — the wider field

## References
- Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986). Learning representations by back-propagating errors. Nature.
- LeCun, Y., Bengio, Y., and Hinton, G. (2015). Deep learning. Nature.
- Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
- Kingma, D. P., and Welling, M. (2014). Auto-encoding variational Bayes. ICLR.
- Whittington, J. C. R., and Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. Neural Computation.
