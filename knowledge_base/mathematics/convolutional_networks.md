---
title: Convolutional Networks
description: Neural networks that apply shared local filters across spatial or temporal input, with pooling and depth, used for vision and compared with the hierarchy of the primate ventral visual stream.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - deep_learning
  - vision
  - neural_networks
semantic_relations:
  - type: relates
    links:
      - "[[neural_networks]]"
      - "[[deep_learning]]"
      - "[[hierarchical_processing]]"
      - "[[ventral_stream]]"
      - "[[feature_extraction]]"
---

# Convolutional Networks

## Overview
A convolutional network is a neural network in which layers apply the same small filter at every position of the input. This builds in two assumptions: local structure, since nearby inputs are more related than distant ones, and translation equivariance, since a feature is useful wherever it occurs. Sharing weights across positions greatly reduces the number of parameters relative to a fully connected layer.

## Structure
For a single-channel image $x$ and a filter $w$, a convolutional layer computes
$$y[i,j]=\sum_{m,n} w[m,n]\,x[i+m,\,j+n]+b,$$
followed by a pointwise nonlinearity (see [[activation_function]]). This is technically a cross-correlation, which is the operation implemented in most software. Layers use many filters in parallel to produce feature maps. Pooling layers summarise small neighbourhoods, for instance by their maximum, giving a degree of invariance to small shifts and reducing resolution. Stacking layers gives units larger receptive fields and increasingly abstract features. Filters are learned by gradient descent with backpropagation.

## History and use
The architecture was anticipated by Fukushima's neocognitron, inspired by the simple and complex cells described by Hubel and Wiesel, and developed for digit recognition by LeCun and colleagues. Deep convolutional networks became dominant in image recognition after the ImageNet results of 2012. Variants are used for audio, time series and other grid-structured data.

## Relation to Active Inference and neuroscience
The strongest link is empirical. Goal-optimised convolutional networks have been used as models of the primate ventral stream, and their intermediate and late layer activations predict neural responses in visual cortex better than many earlier models. This is evidence of a shared hierarchical organisation and not of identical mechanism, since standard networks are feedforward and trained by supervised learning. Predictive coding networks, central to Active Inference accounts of vision, add feedback connections and prediction errors, and convolutional versions of them have been explored as an active research area. Convolutional networks themselves are not derived from the free energy principle.

## Related concepts
- [[ventral_stream]] — the cortical pathway compared with convolutional hierarchies
- [[visual_cortex]] — source of the local receptive field idea
- [[hierarchical_processing]] — layered feature extraction
- [[predictive_coding]] — a feedback-based alternative account of cortical hierarchy
- [[object_recognition]] — a task on which these networks excel
- [[edge_detection]] — a typical low-level feature learned by early filters

## References
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86, 2278-2324.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems, 25, 1097-1105.
- Yamins, D. L. K., & DiCarlo, J. J. (2016). Using goal-driven deep learning models to understand sensory cortex. Nature Neuroscience, 19, 356-365.
