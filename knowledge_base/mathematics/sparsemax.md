---
title: Sparsemax
description: A normalising transformation that maps a score vector to a probability distribution by Euclidean projection onto the simplex, producing exact zeros where softmax gives strictly positive probabilities.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - activation_functions
  - probability_simplex
  - attention
semantic_relations:
  - type: relates
    links:
      - "[[softmax_function]]"
      - "[[probability_simplex]]"
      - "[[attention_mechanisms]]"
      - "[[policy_selection]]"
      - "[[entropy]]"
---

# Sparsemax

## Overview
Sparsemax, introduced by Martins and Astudillo (2016), maps a vector $\mathbf{z}\in\mathbb{R}^K$ to a point of the probability simplex $\Delta^{K-1}$ as the Euclidean projection of $\mathbf{z}$ onto it:
$$\mathrm{sparsemax}(\mathbf{z})=\arg\min_{\mathbf{p}\in\Delta^{K-1}}\lVert\mathbf{p}-\mathbf{z}\rVert_2^2 .$$
Unlike the [[softmax_function]], whose outputs are always strictly positive, sparsemax can assign exactly zero probability to low-scoring entries.

## Closed form
The solution has the thresholded form $p_i=\max(z_i-\tau(\mathbf{z}),0)$. To compute $\tau$, sort the scores as $z_{(1)}\ge\dots\ge z_{(K)}$, let $k(\mathbf{z})=\max\{k: 1+k z_{(k)}>\sum_{j\le k}z_{(j)}\}$, and set
$$\tau(\mathbf{z})=\frac{\sum_{j\le k(\mathbf{z})}z_{(j)}-1}{k(\mathbf{z})}.$$
The cost is dominated by the sort, $O(K\log K)$. Projections onto the simplex of this kind were studied earlier in optimisation, for example by Duchi and colleagues.

## Properties
- **Invariances.** Adding a constant to every score leaves the output unchanged, as for softmax, but scaling the scores changes the support.
- **Sparsity.** Only entries within a margin of the maximum survive. For $K=2$, sparsemax of $(z_1,z_2)$ saturates at $(1,0)$ once $z_1-z_2\ge 1$.
- **Variational view.** Softmax maximises $\langle\mathbf{p},\mathbf{z}\rangle+H(\mathbf{p})$ with Shannon entropy $H$. Sparsemax maximises $\langle\mathbf{p},\mathbf{z}\rangle-\tfrac12\lVert\mathbf{p}\rVert^2$, which corresponds to a quadratic, Tsallis-type entropy with index 2. The family of entmax transformations interpolates between the two.
- **Gradient.** With support set $S$ and indicator vector $\mathbf{s}$, the Jacobian is $\mathrm{diag}(\mathbf{s})-\mathbf{s}\mathbf{s}^{\top}/|S|$. It vanishes outside the support.

## Relation to Active Inference
In discrete Active Inference, the prior over policies is a softmax of negative expected free energy scaled by a precision parameter, so every policy keeps nonzero probability. Replacing it by sparsemax would prune policies outright, which resembles heuristic policy pruning in tree search. This is a design alternative that can be explored in models, not part of the standard formulation, and it breaks the exact link between softmax and the variational posterior over policies. The more common use of sparsemax is in sparse [[attention_mechanisms]].

## Related concepts
- [[softmax_function]] — the dense transformation that sparsemax replaces
- [[probability_simplex]] — the set onto which scores are projected
- [[attention_mechanisms]] — the main application, with sparse attention weights
- [[policy_selection]] — where a softmax over policies appears in Active Inference
- [[entropy]] — the regulariser that distinguishes softmax from sparsemax

## References
- Martins, A. F. T., & Astudillo, R. F. (2016). From softmax to sparsemax: A sparse model of attention and multi-label classification. Proceedings of the 33rd International Conference on Machine Learning, PMLR 48, 1614-1623.
- Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008). Efficient projections onto the l1-ball for learning in high dimensions. Proceedings of the 25th International Conference on Machine Learning, 272-279.
- Tsallis, C. (1988). Possible generalization of Boltzmann-Gibbs statistics. Journal of Statistical Physics, 52, 479-487.
