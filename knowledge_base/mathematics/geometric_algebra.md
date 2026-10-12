---
title: Geometric Algebra
description: Clifford algebra presented through the geometric product, which combines the inner and outer products of vectors and represents rotations, planes and volumes in one associative algebra with applications in physics and computer graphics.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - clifford_algebra
  - geometry
  - rotations
semantic_relations:
  - type: relates
    links:
      - "[[linear_algebra]]"
      - "[[tensor_calculus]]"
      - "[[lie_groups]]"
      - "[[differential_geometry]]"
      - "[[calculus]]"
---

# Geometric Algebra

## Overview
Geometric algebra is the Clifford algebra of a vector space equipped with a quadratic form, developed and advocated as a unified mathematical language for geometry and physics by David Hestenes. Its central operation is the **geometric product**, which for vectors $a$ and $b$ is
$$ab=a\cdot b+a\wedge b,$$
the sum of a scalar (the inner product) and a bivector (the outer product, representing the oriented plane spanned by $a$ and $b$). The product is associative and invertible for non-null vectors. Standard references are Hestenes and Sobczyk (1984), Doran and Lasenby (2003) for physics, and Dorst, Fontijne and Mann (2007) for computer science.

## Structure
In an $n$-dimensional space the algebra has dimension $2^n$ and splits into **grades**: scalars (grade 0), vectors (1), bivectors (2), and so on up to the pseudoscalar of grade $n$. Orthonormal basis vectors satisfy $e_ie_i=\pm1$ according to the signature and $e_ie_j=-e_je_i$ for $i\ne j$. In three Euclidean dimensions the three unit bivectors $e_1e_2$, $e_2e_3$, $e_3e_1$ each square to $-1$, and the pseudoscalar $I=e_1e_2e_3$ squares to $-1$ and commutes with every element.

## Rotations through rotors
A unit bivector $B$ with $B^2=-1$ generates a rotation in its plane. For an angle $\theta$ the **rotor** is
$$R=\exp(-\tfrac12\theta B)=\cos\tfrac{\theta}{2}-B\sin\tfrac{\theta}{2},$$
and a vector $v$ is rotated by $v\mapsto R\,v\,\tilde R$, where $\tilde R$ is the reverse of $R$. The even subalgebra of the three-dimensional algebra is isomorphic to the quaternions, so rotors reproduce quaternion rotations. The same formula works in any dimension and signature, and the rotors form the spin group, a double cover of the rotation group (see [[lie_groups]]).

## Geometric calculus and physics
Extending the algebra with a vector derivative $\nabla$ gives geometric calculus, in which divergence and curl are combined as the two parts of $\nabla F=\nabla\cdot F+\nabla\wedge F$, and the fundamental theorem of geometric calculus generalises Stokes' theorem (Hestenes and Sobczyk, 1984). Applied to Minkowski spacetime, the spacetime algebra rewrites the four Maxwell equations as a single equation in the field bivector. Hestenes (2003) argued for reforming the mathematical language of physics in this direction.

## Computational uses
In computer graphics and robotics, geometric algebra provides a uniform representation of points, lines, planes, spheres and rigid motions, and its conformal variant treats translations as rotors (Dorst et al., 2007).

## Relation to Active Inference
There is no established connection. Active Inference is formulated with probability and, in some treatments, differential geometry (see [[differential_geometry]]), and does not use Clifford algebra. Embodied-agent models that must represent orientations could in principle use rotors in place of rotation matrices or quaternions, but this would be a representation choice and not part of the framework.

## Related concepts
- [[linear_algebra]] — the vector space structure on which the algebra is built
- [[tensor_calculus]] — an alternative language for multilinear geometry
- [[lie_groups]] — rotation groups and their covers
- [[differential_geometry]] — calculus on manifolds
- [[calculus]] — the calculus that geometric calculus unifies and extends

## References
- Hestenes, D., & Sobczyk, G. (1984). Clifford Algebra to Geometric Calculus: A Unified Language for Mathematics and Physics. D. Reidel.
- Doran, C., & Lasenby, A. (2003). Geometric Algebra for Physicists. Cambridge University Press. https://doi.org/10.1017/CBO9780511807497
- Hestenes, D. (2003). Oersted Medal Lecture 2002: Reforming the mathematical language of physics. American Journal of Physics, 71(2), 104-121. https://doi.org/10.1119/1.1522700
- Dorst, L., Fontijne, D., & Mann, S. (2007). Geometric Algebra for Computer Science: An Object-Oriented Approach to Geometry. Morgan Kaufmann.
