---
title: State Space
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - state_space
  - dynamical_systems
  - control_theory
semantic_relations:
  - type: foundation
    links:
      - "[[dynamical_systems]]"
      - "[[control_theory]]"
  - type: relates
    links:
      - "[[state_spaces]]"
      - "[[state_space_abstraction]]"
      - "[[active_inference_pomdp]]"
---

# State Space

## Overview
A state space is the set of all configurations a system can be in, such that the state at one time together with the inputs determines (or, in a stochastic setting, parametrises the distribution of) its future. State-space models describe a system by a hidden state evolving in time and an observation produced from it. For the design of state spaces inside Active Inference generative models see [[state_spaces]] and [[state_space_abstraction]]; this page gives the general mathematical definition.

## Definition / Formalism
Continuous time: $\dot x = f(x,u)$, $y=h(x,u)$, with $x\in\mathbb R^n$ the state, $u$ the input and $y$ the output. Discrete time with noise: $x_{t+1}=f(x_t,u_t)+w_t$, $y_t=h(x_t)+v_t$. Equivalent probabilistic form: a transition $p(x_{t+1}\mid x_t,u_t)$ and an observation model $p(y_t\mid x_t)$, which makes the model a hidden Markov model for discrete $x$ and a POMDP when actions are chosen.

For linear time-invariant systems $\dot x=Ax+Bu$, $y=Cx$: the system is controllable iff $[B,\,AB,\,\dots,\,A^{n-1}B]$ has rank $n$, and observable iff the stacked matrix $[C;\,CA;\,\dots;\,CA^{n-1}]$ has rank $n$. With Gaussian noise and linear maps, exact filtering of the hidden state is given by the Kalman filter. In classical mechanics the state space of a Hamiltonian system is its phase space of positions and momenta, a symplectic manifold. The state space need not be Euclidean: it can be discrete, a manifold, or a space of probability distributions (a belief space).

## Role in Active Inference
Generative models in active inference are state-space models. In the discrete case the hidden-state space is a finite set (possibly factorised into several factors), $B$ encodes transitions between states and $A$ maps states to observations. In continuous time the state is augmented to generalized coordinates, a stack of temporal derivatives. A less obvious point is that the agent's beliefs form their own space: the posterior $q(s)$ lives on a statistical manifold, and belief updating is a trajectory there, which is where information geometry enters. The agent's state space is its model of the world, not the world's true state space; the two can differ in dimension and structure.

## Related concepts
- [[state_spaces]] — design and analysis of state spaces for generative models
- [[state_space_abstraction]] — coarse-graining and factorisation of state spaces
- [[dynamical_systems]] — evolution of states over time
- [[control_theory]] — controllability, observability and feedback in state-space form
- [[generalized_coordinates]] — extended state for continuous-time models
- [[information_geometry]] — geometry of the space of beliefs
