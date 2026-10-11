---
title: Probability Simplex
description: The set of probability vectors over a finite number of outcomes, its geometry as a convex polytope and Riemannian manifold, and its role as the space of categorical beliefs in discrete Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - probability
  - geometry
  - categorical_distributions
semantic_relations:
  - type: relates
    links:
      - "[[action_distribution]]"
      - "[[softmax_function]]"
      - "[[belief_state]]"
      - "[[statistical_manifolds]]"
      - "[[fisher_information]]"
      - "[[information_geometry]]"
---

# Probability Simplex

## Overview
The probability simplex over $n$ outcomes is the set of all probability vectors
$$\Delta^{n-1}=\Big\{p\in\mathbb{R}^n:\; p_i\ge0,\;\sum_{i=1}^n p_i=1\Big\}.$$
Every categorical distribution over $n$ categories is a point of this set. Its dimension is $n-1$, because the normalisation constraint removes one degree of freedom. For $n=2$ it is a line segment, for $n=3$ a triangle, and for $n=4$ a tetrahedron.

## Geometry
As a subset of Euclidean space the simplex is a convex polytope with $n$ vertices, the one-hot distributions concentrating all mass on one outcome. Mixtures of distributions stay in the set, and the uniform distribution lies at the centroid. The boundary consists of distributions with at least one zero probability, and the interior contains those with full support.

The Fisher information metric, $ds^2=\sum_i dp_i^2/p_i$, turns the interior into a Riemannian manifold. The map $x_i=2\sqrt{p_i}$ is an isometry onto the positive orthant of a sphere of radius 2, so geodesic distances are arcs of great circles. This is the geometric setting for information geometry of categorical models, where the Kullback-Leibler divergence plays the role of a generalised squared distance.

## Parameterisations
Unconstrained optimisation on the simplex typically uses a coordinate change. The softmax function maps $\mathbb{R}^n$ onto the interior and makes gradient updates on logits automatically satisfy non-negativity and normalisation. Dirichlet distributions place a density on the simplex and are conjugate to categorical likelihoods. Constrained algorithms such as projected gradient and mirror descent with entropic regularisation, whose iterates are multiplicative updates, operate directly on the set.

## Relation to Active Inference
The simplex is where discrete beliefs live. Beliefs over hidden states, the columns of likelihood and transition arrays, the posterior over policies and the distribution over actions are all points of simplices. A factorised model uses a product of simplices, one for each factor. Belief updating in the discrete formulation can therefore be viewed as movement over this space, and free energy as a function on it, with softmax operations ensuring that updates remain valid distributions. This is a direct mathematical identification, not an analogy.

## Related concepts
- [[action_distribution]] — a distribution over actions, a point on a simplex
- [[softmax_function]] — a map from real vectors to the interior of the simplex
- [[belief_state]] — a distribution over hidden states
- [[statistical_manifolds]] — spaces of distributions with Fisher geometry
- [[fisher_metric]] — the Riemannian metric on the interior
- [[information_geometry]] — the field that studies this geometry

## References
- Amari, S., & Nagaoka, H. (2000). Methods of Information Geometry. American Mathematical Society and Oxford University Press.
- Boyd, S., & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.
- Ay, N., Jost, J., Lê, H. V., & Schwachhöfer, L. (2017). Information Geometry. Springer.
