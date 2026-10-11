---
title: Limit Cycles
description: Isolated closed trajectories of nonlinear dynamical systems that attract or repel nearby states, their detection, standard examples, and their relevance to biological oscillation and free energy dynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - dynamical_systems
  - oscillations
  - nonlinear_dynamics
semantic_relations:
  - type: relates
    links:
      - "[[dynamical_systems]]"
      - "[[attractor_dynamics]]"
      - "[[bifurcation_theory]]"
      - "[[hopf_bifurcation]]"
      - "[[lyapunov_stability]]"
      - "[[chaos_theory]]"
---

# Limit Cycles

## Overview
A limit cycle is an isolated closed orbit of an autonomous dynamical system: a periodic trajectory in phase space with no other periodic trajectory arbitrarily close to it. It is stable if nearby trajectories converge to it as time increases, unstable if they diverge, and semi-stable if they approach from one side only. A stable limit cycle is a self-sustained oscillation. Its amplitude and period are set by the system, not by the initial condition.

## Properties and examples
Linear systems cannot have limit cycles. Their periodic solutions come in continuous families whose amplitude depends on the initial state. Limit cycles therefore require nonlinearity. The classic example is the van der Pol oscillator, $\ddot{x}-\mu(1-x^2)\dot{x}+x=0$ with $\mu>0$, which has a single stable limit cycle. In the plane, the Poincaré-Bendixson theorem states that a bounded trajectory that remains away from fixed points tends to a closed orbit, so planar flows cannot be chaotic. In three or more dimensions, more complicated behaviour, including chaos, becomes possible.

Limit cycles are often born at a Hopf bifurcation, when a fixed point loses stability and a small oscillation appears. The stability of a cycle can be assessed through its Floquet multipliers, the eigenvalues of the linearised return map on a cross-section.

## Occurrence in biology
Heartbeat, circadian rhythms, the cell cycle, spiking in neurons driven by constant current, and some predator-prey dynamics have been modelled with limit cycles. Models of coupled oscillators, in which cycles synchronise, are used for neural rhythms and for collective rhythms in insect colonies.

## Relation to Active Inference
Free energy minimisation is often stated in terms of attracting sets. A limit cycle is one kind of attractor, and it is the natural description of an agent whose characteristic states are rhythmic rather than static. The free energy literature allows non-equilibrium steady states with solenoidal flow, which circulates around level sets of the density and can produce cyclic behaviour. Models of sequences and rhythms in perception and action, including those built from itinerant or heteroclinic dynamics, are in this spirit. The connection is a modelling choice and an analogy for specific systems, not a result that every self-organising system follows a limit cycle.

## Related concepts
- [[dynamical_systems]] — the theory in which limit cycles are defined
- [[attractor_dynamics]] — limit cycles as one type of attractor
- [[hopf_bifurcation]] — a common route by which cycles appear
- [[synchronization]] — coupling of oscillators on limit cycles
- [[lyapunov_stability]] — stability notions applied to periodic orbits

## References
- Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos (2nd ed.). Westview Press.
- Perko, L. (2001). Differential Equations and Dynamical Systems (3rd ed.). Springer.
- Glass, L., & Mackey, M. C. (1988). From Clocks to Chaos: The Rhythms of Life. Princeton University Press.
