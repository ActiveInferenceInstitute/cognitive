---
title: Lie Groups
description: "Smooth manifolds that are also groups, with matrix examples such as SO(3) and SE(3), group actions, invariance, and limited uses in geometric treatments of inference."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - mathematics
  - groups
  - geometry
semantic_relations:
  - type: relates
    links:
      - "[[lie_theory]]"
      - "[[differential_geometry]]"
      - "[[geometric_mechanics]]"
      - "[[information_geometry]]"
---

# Lie Groups

## Overview
A Lie group is a group $G$ that is also a smooth manifold, with multiplication and inversion smooth maps. The tangent space at the identity is its Lie algebra (see [[lie_theory]]). Lie groups model continuous families of symmetries such as rotations, translations and scalings.

## Definition and examples
**Matrix Lie groups.** Closed subgroups of $GL(n)$ are Lie groups (Cartan's closed-subgroup theorem). Standard ones: $GL(n,\mathbb R)$, the orthogonal group $O(n)$ and special orthogonal $SO(n)$ (dimension $n(n-1)/2$, so 3 for $SO(3)$), the unitary groups $U(n)$ and $SU(n)$, and the special Euclidean group $SE(n)=SO(n)\ltimes\mathbb R^n$ of rigid motions ($SE(3)$ has dimension 6).

**Properties.** Compact connected groups such as $SO(3)$ admit a bi-invariant Riemannian metric and an invariant (Haar) measure, allowing averaging over the group. $SO(3)$ is not simply connected; its universal cover is $SU(2)$, a double cover.

**Actions and homogeneous spaces.** A Lie group acts on a manifold $M$ smoothly; the orbit of a point is $G/H$ where $H$ is the stabiliser. A function is invariant if constant on orbits and equivariant if it commutes with the action. Spheres are $SO(n+1)/SO(n)$; the Gaussian family with fixed covariance is acted on transitively by translations.

**Left-invariant dynamics.** On $G$, $\dot g=g\,\xi$ with $\xi\in\mathfrak g$ gives the flow $g(t)=g(0)\exp(t\xi)$. Rigid-body mechanics on $SO(3)$ and Euler-Poincare reduction are the standard application ([[geometric_mechanics]]).

## Role in Active Inference
Direct use is limited and specialised. Three connections are concrete: (1) pose estimation and motor control for embodied agents work on $SE(3)$ and $SO(3)$, so a generative model of body pose has Lie group state space and requires beliefs on a manifold rather than $\mathbb R^n$; (2) in [[information_geometry]] the affine group acts on families of distributions, and the Fisher metric is invariant under it; (3) symmetry arguments in [[free_energy_principle]] treatments use group actions to relate conserved quantities to the flow decomposition. These are modelling tools; the discrete Active Inference scheme uses none of them.

## Related concepts
- [[lie_theory]] — the Lie algebra, exponential map and classification
- [[differential_geometry]] — manifolds, tangent bundles, Riemannian metrics
- [[symplectic_geometry]] — momentum maps for Hamiltonian group actions
- [[poisson_geometry]] — Lie-Poisson structure on duals of Lie algebras
- [[statistical_manifolds]] — group actions on parametric families
