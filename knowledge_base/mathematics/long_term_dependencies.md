---
title: Long-Term Dependencies
description: The difficulty of learning relationships between events separated by many time steps in sequence models, caused by vanishing and exploding gradients, and the gated and attention-based remedies.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - sequence_models
  - recurrent_networks
  - gradients
semantic_relations:
  - type: relates
    links:
      - "[[recurrent_networks]]"
      - "[[temporal_models]]"
      - "[[attention_mechanisms]]"
      - "[[working_memory]]"
      - "[[automatic_differentiation]]"
---

# Long-Term Dependencies

## Overview
A long-term dependency exists when the correct output at one time step depends on input seen many steps earlier, such as the subject of a sentence determining a verb far downstream. Learning such relationships by gradient descent is hard in recurrent networks, and Bengio, Simard and Frasconi (1994) showed that this difficulty is intrinsic to training with gradients rather than an artefact of a particular architecture.

## The gradient problem
Consider a recurrent network with hidden state $h_t=f(Wh_{t-1}+Ux_t)$. Backpropagation through time gives
$$\frac{\partial h_T}{\partial h_t}=\prod_{k=t+1}^{T}\mathrm{diag}\big(f'(a_k)\big)\,W,$$
where $a_k$ is the pre-activation at step $k$. If the largest singular value of $W$ multiplied by a bound on $|f'|$ is below one, the norm of the product shrinks geometrically in $T-t$ and the gradient vanishes, so the influence of distant inputs cannot be learned. If the spectral radius of $W$ is large enough, the product can grow geometrically and gradients explode. Bengio and colleagues further argued that a network which latches information robustly, by remaining in a stable attractor, is exactly the regime in which gradients with respect to earlier states shrink.

## Remedies
- **Gating.** Long short-term memory (Hochreiter and Schmidhuber, 1997) uses a memory cell with an additive update and multiplicative gates, creating a path along which error flows without repeated squashing. Gated recurrent units are a simpler variant.
- **Gradient clipping** bounds exploding gradients, and careful initialisation of $W$, for example near orthogonal, slows both effects.
- **Attention and skip connections** give direct paths between distant positions, so the dependency length seen by the gradient is short. This is a main reason transformer models handle long contexts (see [[attention_mechanisms]]).

## Relation to Active Inference
Hierarchical generative models in Active Inference treat long-range temporal structure by giving higher levels slower timescales, so that a slowly changing context constrains many fast transitions below. This is a modelling choice with a different mechanism from gating, although both address how information is carried across time. The comparison is an analogy and not a derivation.

## Related concepts
- [[recurrent_networks]] — the architectures in which the problem arises
- [[temporal_models]] — models of sequences and time in cognition
- [[attention_mechanisms]] — direct access to earlier positions
- [[working_memory]] — biological maintenance of information over time
- [[language_modeling]] — a task where long-range dependencies matter
- [[automatic_differentiation]] — the mechanism computing the gradient products

## References
- Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 5, 157-166.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9, 1735-1780.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
