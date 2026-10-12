---
title: Gauge Equivariant Convolutional Networks
description: Convolutional neural networks on manifolds whose layers are equivariant to local changes of reference frame, so that their output does not depend on the arbitrary choice of coordinates used to express kernels and features.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - geometric_deep_learning
  - equivariance
  - fiber_bundles
semantic_relations:
  - type: relates
    links:
      - "[[convolutional_networks]]"
      - "[[fiber_bundles]]"
      - "[[lie_groups]]"
      - "[[differential_geometry]]"
      - "[[neural_network_theory]]"
---

# Gauge Equivariant Convolutional Networks

## Overview
A convolution on a flat grid applies the same kernel at every position, and this weight sharing is justified by translation symmetry. On a curved surface or a general manifold there is no global translation symmetry, and, more fundamentally, there is no canonical way to orient a kernel: at each point one must choose a reference frame (a gauge), and different choices rotate the kernel relative to the data. Gauge equivariant CNNs, introduced by Cohen, Weiler, Kicanaoglu and Welling (2019), require that a layer's output transform in a prescribed way under local changes of frame, so that the network depends only on the intrinsic geometry of the manifold.

## Core ideas
**Features are sections of a bundle.** On a manifold $M$ with a $G$-structure (for a surface, $G=\mathrm{SO}(2)$ rotations of the tangent plane), a feature field of type $\rho$ is a section of the associated vector bundle built from the frame bundle and a representation $\rho$ of $G$. In a given gauge it is a vector-valued function whose components change by $\rho(g)$ when the frame at a point is changed by $g\in G$. Scalars, tangent vectors and higher tensor fields correspond to different choices of $\rho$.

**Kernel constraint.** If a layer maps fields of type $\rho_{\mathrm{in}}$ to type $\rho_{\mathrm{out}}$, a kernel $K$ defined on a tangent plane is consistent across gauges when
$$K(g\cdot v)=\rho_{\mathrm{out}}(g)\,K(v)\,\rho_{\mathrm{in}}(g)^{-1}\qquad\text{for all }g\in G,$$
the same steerability condition used in equivariant networks on flat space. The constraint is what allows weight sharing without a preferred orientation. Neighbouring features are compared by transporting them along the manifold, which is where the geometry enters. Weiler, Forré, Verlinde and Welling (2021) show that simultaneous demands for coordinate independence and weight sharing lead exactly to this requirement of gauge equivariance, and they express the theory in terms of fibre bundles (see [[fiber_bundles]]).

## Relation to other equivariant networks
Group equivariant CNNs (Cohen and Welling, 2016) build in global symmetries such as rotations and translations of the plane. Gauge equivariance extends the principle from global symmetries to local gauge transformations and includes many existing methods as special cases. Bronstein, Bruna, Cohen and Veličković (2021) treat gauges as one of five organising notions of geometric deep learning, alongside grids, groups, graphs and geodesics.

## Example
For signals on a sphere, Cohen et al. (2019) work on the surface of an icosahedron as an approximation to the sphere. This regular mesh lets the gauge equivariant convolution be implemented with a single standard two-dimensional convolution call. They report improvements over earlier spherical methods on segmentation of omnidirectional images and of global climate patterns.

## Relation to Active Inference
There is no established connection. Both areas use symmetry to constrain a model, and geometric treatments of Active Inference work with manifolds of beliefs (see [[lie_groups]] and [[differential_geometry]]). Gauge equivariant networks, however, are a tool for processing signals on a given manifold, not a statement about inference, and any use of them inside an Active Inference generative model would be a design choice.

## Related concepts
- [[convolutional_networks]] — the Euclidean architecture being generalised
- [[fiber_bundles]] — frame bundles and associated bundles
- [[lie_groups]] — structure groups such as $\mathrm{SO}(2)$
- [[differential_geometry]] — connections and parallel transport
- [[neural_network_theory]] — general theory of network architectures

## References
- Cohen, T. S., & Welling, M. (2016). Group equivariant convolutional networks. Proceedings of the 33rd International Conference on Machine Learning, PMLR 48, 2990-2999.
- Cohen, T. S., Weiler, M., Kicanaoglu, B., & Welling, M. (2019). Gauge equivariant convolutional networks and the icosahedral CNN. Proceedings of the 36th International Conference on Machine Learning, PMLR 97, 1321-1330.
- Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv:2104.13478.
- Weiler, M., Forré, P., Verlinde, E., & Welling, M. (2021). Coordinate independent convolutional networks: Isometry and gauge equivariant convolutions on Riemannian manifolds. arXiv:2106.06020.
