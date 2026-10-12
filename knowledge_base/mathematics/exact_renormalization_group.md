---
title: Exact Renormalization Group
description: Formulations of the renormalization group as exact functional differential equations in a continuous scale parameter, including the Wegner-Houghton, Polchinski and Wetterich flow equations, and the truncations needed to use them.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - renormalization
  - statistical_physics
  - quantum_field_theory
  - functional_methods
aliases:
  - Exact Renormalization Group
  - ERG
  - functional renormalization group
semantic_relations:
  - type: relates
    links:
      - "[[renormalization_group]]"
      - "[[bayesian_renormalization]]"
      - "[[coarse_graining]]"
      - "[[statistical_physics]]"
      - "[[quantum_field_theory]]"
      - "[[scale_invariance]]"
---

# Exact Renormalization Group

## Overview
The renormalization group (see [[renormalization_group]]) describes how the effective description of a system changes as short-distance or high-momentum degrees of freedom are removed. In Wilson's formulation the removal is done in finite steps. The exact renormalization group (ERG) replaces the steps with a continuous scale parameter and writes the change of the effective action as an exact functional differential equation. "Exact" refers to the flow equation itself, which contains no small parameter and no truncation. Solving it for a realistic theory generally requires approximations.

## Three standard forms
- **Wegner-Houghton equation.** Wegner and Houghton (1973) derived an exact equation for the Wilsonian action with a sharp momentum cutoff, in which lowering the cutoff integrates out an infinitesimal shell of modes.
- **Polchinski equation.** Polchinski (1984) wrote the flow for the action with a smooth momentum cutoff. The change of the interaction part of the action with the cutoff $\Lambda$ is given by a functional expression built from its first and second functional derivatives, and the form is well suited to arguments about perturbative renormalisability. The Wetterich equation below is related to it by a Legendre transformation.
- **Wetterich equation.** Wetterich (1993) gave a flow for the effective average action $\Gamma_k$, a scale-dependent version of the 1PI effective action in which modes below a scale $k$ are suppressed by a regulator $R_k$. With $t=\ln k$,
$$\partial_t\Gamma_k=\tfrac12\,\mathrm{Tr}\!\left[\left(\Gamma_k^{(2)}+R_k\right)^{-1}\partial_t R_k\right],$$
where $\Gamma_k^{(2)}$ is the second functional derivative of $\Gamma_k$. As $k$ is lowered from the microscopic scale to zero, $\Gamma_k$ interpolates between the bare action and the full effective action (Berges, Tetradis and Wetterich, 2002).

## Using the equations
Because each flow equation is an infinite tower of coupled equations for the couplings, it is solved by truncating the space of actions, for example to a derivative expansion with a few terms. Fixed points of the rescaled flow describe scale-invariant behaviour and critical points, and the linearised flow around them yields critical exponents. The scale evolution is a semigroup, not a group, because coarse-graining discards information and cannot be undone (see [[coarse_graining]]).

## Relation to Active Inference
The connection is an analogy that has been proposed in the literature and should not be read as established. A flow equation for probability distributions or couplings under change of scale has the same mathematical shape as some proposals for treating inference as a flow in a space of distributions, which is the idea behind [[bayesian_renormalization]]. The free-energy principle uses nested Markov blankets to describe separation of scales, but it does not use the ERG equations themselves, and no derivation of one from the other is established.

## References
- Wegner, F. J., & Houghton, A. (1973). Renormalization group equation for critical phenomena. Physical Review A, 8(1), 401-412. doi:10.1103/physreva.8.401
- Polchinski, J. (1984). Renormalization and effective Lagrangians. Nuclear Physics B, 231(2), 269-295. doi:10.1016/0550-3213(84)90287-6
- Wetterich, C. (1993). Exact evolution equation for the effective potential. Physics Letters B, 301(1), 90-94. doi:10.1016/0370-2693(93)90726-x
- Berges, J., Tetradis, N., & Wetterich, C. (2002). Non-perturbative renormalization flow in quantum field theory and statistical physics. Physics Reports, 363(4-6), 223-386. doi:10.1016/s0370-1573(01)00098-9
