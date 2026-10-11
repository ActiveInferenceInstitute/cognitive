---
title: Transformer Networks
description: Neural architectures built on self-attention instead of recurrence or convolution, covering scaled dot-product attention, multi-head structure, and how they relate to cognitive attention and predictive processing.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - deep_learning
  - attention
  - sequence_models
semantic_relations:
  - type: relates
    links:
      - "[[attention_mechanisms]]"
      - "[[neural_networks]]"
      - "[[recurrent_networks]]"
      - "[[softmax_function]]"
      - "[[hopfield_networks]]"
---

# Transformer Networks

## Overview
A transformer is a neural network architecture for sequences and sets in which each element is updated by attending to all others. Introduced for machine translation, it replaced recurrence with attention, allowing every position in a sequence to be processed in parallel during training. Transformers are the basis of most current large language models.

## Mechanism
Each input vector is linearly projected into a query, a key and a value. Scaled dot-product attention computes
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,$$
so each output is a weighted average of value vectors, with weights given by the similarity of its query to the keys. Several attention operations run in parallel as heads with separate projections, and their outputs are concatenated. A transformer block combines this with a position-wise feed-forward network, residual connections and layer normalisation. Because attention is symmetric under permutation of positions, order must be injected by positional encodings. The cost of full self-attention grows quadratically with sequence length, which motivates efficient variants. Language models typically use a causal mask so that each position attends only to earlier ones and is trained to predict the next token.

## Relation to cognition and Active Inference
The word attention is shared with psychology, but the senses differ: here it is a learned, differentiable weighting of stored vectors, with no assumption of a limited-capacity resource. Within predictive processing, attention is usually cast as precision weighting, the modulation of the gain on prediction errors, and a formal mapping between that and softmax attention has not been established. A better-founded connection is mathematical. The update rule of modern continuous Hopfield networks is equivalent to a form of attention, which links transformers to associative memory. Autoregressive training minimises a next-token prediction error, which resembles prediction-based learning in a loose sense, but transformers are trained by backpropagation on a likelihood, not by free energy minimisation. These links are analogies and active research questions.

## Related concepts
- [[attention_mechanisms]] — cognitive and computational senses of attention
- [[recurrent_networks]] — the sequence models that attention largely replaced
- [[hopfield_networks]] — associative memory networks whose update resembles attention
- [[softmax_function]] — the normalisation producing attention weights
- [[precision_weighting]] — the predictive processing account of attention
- [[deep_learning]] — the wider field

## References
- Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30, 5998-6008.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
