---
title: Computational Fluid Dynamics
description: The numerical solution of the equations of fluid motion on a computer, covering the Navier-Stokes equations, discretisation methods, turbulence treatment and verification, with applications to biological flows such as insect flight.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - fluid_dynamics
  - numerical_methods
  - simulation
  - biomechanics
aliases:
  - CFD
semantic_relations:
  - type: relates
    links:
      - "[[numerical_methods]]"
      - "[[differential_equations]]"
      - "[[numerical_stability]]"
      - "[[insect_flight_dynamics]]"
      - "[[biomechanics]]"
      - "[[digital_twins]]"
---

# Computational Fluid Dynamics

## Overview
Computational fluid dynamics (CFD) is the use of numerical methods to compute the motion of liquids and gases, together with the forces they exert on bodies, by solving the governing equations of fluid mechanics on a discretised domain. It substitutes simulation for analytical solution in the many flows, such as those around a flapping wing, for which no closed-form solution exists, and complements wind-tunnel and flume experiments.

## Governing equations
For an incompressible Newtonian fluid of constant density $\rho$ and kinematic viscosity $\nu$, the velocity field $\mathbf{u}(\mathbf{x},t)$ and pressure $p$ obey the Navier-Stokes equations
$$\frac{\partial\mathbf{u}}{\partial t}+(\mathbf{u}\cdot\nabla)\mathbf{u}=-\frac{1}{\rho}\nabla p+\nu\nabla^2\mathbf{u}+\mathbf{f},\qquad \nabla\cdot\mathbf{u}=0,$$
with body force $\mathbf{f}$. The dimensionless Reynolds number $\mathrm{Re}=UL/\nu$, built from a characteristic speed $U$ and length $L$, measures the ratio of inertial to viscous effects and decides which terms dominate and whether the flow is steady, unsteady or turbulent.

## Methods
- **Discretisation.** Finite-difference, finite-volume and finite-element methods turn the partial differential equations into algebraic equations on a grid or mesh. Finite-volume schemes enforce conservation of mass and momentum over each cell and are widely used in engineering codes.
- **Time stepping and pressure coupling.** Incompressibility couples velocity and pressure, so solvers use projection or pressure-correction steps, and choice of scheme affects stability (see [[numerical_stability]]).
- **Turbulence.** Direct numerical simulation resolves all scales but is expensive. Large-eddy simulation resolves the large scales and models the small ones, and Reynolds-averaged methods model the effect of all turbulent fluctuations.
- **Moving boundaries.** Flapping or swimming bodies require moving or overlapping meshes, or immersed-boundary treatments in which the body is represented on a fixed grid.
- **Verification and validation.** Results should be checked against grid refinement, conservation properties and experimental data, since a converged solution of a modelling error is still wrong.

## Application to insect flight
Liu and colleagues (1998) used CFD to study a virtual wing modelled on a hovering hawkmoth, and found a coherent leading-edge vortex during the translational phases of the stroke that enhances lift. Wang (2005) reviews the unsteady aerodynamics of flapping flight at intermediate Reynolds numbers and the mechanisms behind force production in hovering insects; Sane (2003) also treats the aerodynamics of insect flight. Simulations of this kind supply force estimates and flow structure that are hard to measure directly (see [[insect_flight_dynamics]]).

## Relation to Active Inference
The connection is indirect. CFD is a tool for simulating the physics of an agent's environment or body, for example to build a high-fidelity simulator or digital twin (see [[digital_twins]]), against which a controller or generative model can be tested. Active Inference itself does not require fluid dynamics. A fluid solver used this way belongs to the world model of the experimenter, not to the agent's generative model.

## References
- Ferziger, J. H., & Perić, M. (1996). Computational Methods for Fluid Dynamics. Springer.
- Liu, H., Ellington, C. P., Kawachi, K., Van Den Berg, C., & Willmott, A. P. (1998). A computational fluid dynamic study of hawkmoth hovering. Journal of Experimental Biology, 201(4), 461-477. doi:10.1242/jeb.201.4.461
- Wang, Z. J. (2005). Dissecting insect flight. Annual Review of Fluid Mechanics, 37, 183-210. doi:10.1146/annurev.fluid.36.050802.121940
- Sane, S. P. (2003). The aerodynamics of insect flight. Journal of Experimental Biology, 206(23), 4191-4208. doi:10.1242/jeb.00663
