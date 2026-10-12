---
title: Deep Equilibrium Models
description: Deep equilibrium models define a network output as the fixed point of a single layer, solved by root finding and differentiated through the implicit function theorem with memory independent of depth.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - deep_learning
  - fixed_points
  - implicit_models
semantic_relations:
  - type: relates
    links:
      - "[[neural_odes]]"
      - "[[attractor_dynamics]]"
      - "[[contraction]]"
      - "[[neural_network_theory]]"
      - "[[gradient_descent]]"
---

# Deep Equilibrium Models

## Overview
A deep equilibrium model (DEQ) is a neural network whose output is defined implicitly, as the equilibrium of a layer applied repeatedly to its own output. Instead of stacking many distinct layers, a DEQ uses one layer $f_\theta$ with input injection $x$ and takes the limit of infinite depth, so the representation $z^\*$ satisfies
$$z^\*=f_\theta(z^\*;x).$$
Bai, Kolter and Koltun proposed the construction for sequence models and showed that weight-tied, input-injected networks could reach the performance of much deeper explicit models.

## Forward and backward computation
The forward pass does not iterate a fixed number of layers. It applies a root-finding method, such as Broyden's quasi-Newton method or Anderson acceleration, to $g_\theta(z)=f_\theta(z;x)-z=0$. Any solver can be used, and its accuracy can be traded against cost at test time.

The backward pass uses the implicit function theorem. If $I-\partial f_\theta/\partial z$ is invertible at $z^\*$, then
$$\frac{\partial z^\*}{\partial \theta}=\left(I-\frac{\partial f_\theta}{\partial z}\bigg|_{z^\*}\right)^{-1}\frac{\partial f_\theta}{\partial \theta}\bigg|_{z^\*}.$$
In practice the inverse is not formed; the vector-Jacobian product needed for backpropagation is obtained by solving a second linear fixed-point problem. The consequence is that no intermediate activations of the forward solver are stored, so memory cost is constant in the effective depth.

## Well-posedness
Existence and uniqueness of $z^\*$ are not guaranteed for arbitrary $f_\theta$. If $f_\theta$ is a contraction in $z$, the Banach fixed-point theorem gives both, and plain iteration converges linearly. Without such a constraint the solver can diverge or find one of several equilibria. Later work restricts the parameterisation, for instance with monotone-operator layers, so that a unique equilibrium exists by construction. Related implicit-layer ideas, including neural ODEs and optimisation layers, share the use of implicit differentiation.

## Relation to Active Inference
Perception in predictive-coding and Active Inference schemes is commonly written as a gradient flow on free energy whose steady state is the posterior mode, and so is itself an equilibrium of recurrent dynamics. A DEQ resembles this at the level of computation, because the output is the settled state of an iterated update. The resemblance is an analogy. DEQs are trained by supervised losses and do not minimise variational free energy unless they are explicitly built to do so.

## Related concepts
- [[neural_odes]] — another implicit, continuous-depth architecture
- [[attractor_dynamics]] — equilibria as stable states of a dynamical system
- [[contraction]] — the condition guaranteeing a unique fixed point
- [[neural_network_theory]] — theoretical treatment of network architectures
- [[taylor_series]] — local expansions used to analyse and approximate implicit maps
- [[gradient_descent]] — the optimisation used to train the layer parameters

## References
- Bai, S., Kolter, J. Z., & Koltun, V. (2019). Deep equilibrium models. Advances in Neural Information Processing Systems, 32.
- Winston, E., & Kolter, J. Z. (2020). Monotone operator equilibrium networks. Advances in Neural Information Processing Systems, 33.
- Krantz, S. G., & Parks, H. R. (2002). The Implicit Function Theorem: History, Theory, and Applications. Birkhauser.
