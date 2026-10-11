---
title: Reaction-Diffusion Systems
description: Partial differential equations combining local reactions with spatial diffusion, Turing's diffusion-driven instability, resulting biological patterns, and links to morphogenesis modelled as inference under the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - pattern_formation
  - morphogenesis
  - partial_differential_equations
semantic_relations:
  - type: relates
    links:
      - "[[pattern_formation]]"
      - "[[morphogenesis]]"
      - "[[developmental_systems]]"
      - "[[diffusion_processes]]"
      - "[[bifurcation_theory]]"
      - "[[friston_2015_knowing]]"
---

# Reaction-Diffusion Systems

## Overview
A reaction-diffusion system describes how the concentrations of one or more substances change in space and time through two processes: local reactions that create or consume the substances, and diffusion that spreads them. For a vector of concentrations $u(x,t)$ the general form is
$$\partial_t u=D\,\nabla^2u+f(u),$$
where $D$ is a diagonal matrix of diffusion coefficients and $f$ collects the reaction terms. Such equations model chemical waves, population invasions, excitable tissue and pattern formation.

## Turing instability
In 1952 Alan Turing showed that diffusion can destabilise a state that is stable without it. Consider two species, an activator and an inhibitor, with a homogeneous steady state that is stable to spatially uniform perturbations. If the inhibitor diffuses sufficiently faster than the activator, perturbations of intermediate spatial wavelength grow, and the homogeneous state gives way to a stationary pattern with a characteristic wavelength set by the reaction rates and diffusion coefficients. Linear stability analysis expresses this through the dispersion relation, in which the growth rate becomes positive for a band of wavenumbers. Nonlinear terms then saturate the growth into spots, stripes or labyrinthine patterns.

## Models and applications
Standard models include the activator-inhibitor system of Gierer and Meinhardt, the Gray-Scott system for self-replicating spots, and the Fisher-KPP equation for travelling fronts of an advantageous allele or an invasive species. Proposed biological applications include animal coat patterns, digit spacing and skin appendage arrangements. Evidence for Turing mechanisms in specific tissues varies, and some patterns are explained by competing mechanisms such as mechanical or cell-signalling models.

## Relation to Active Inference
The connection is an active research programme on morphogenesis. Friston and colleagues modelled a collective of cells that each minimise free energy under a shared generative model of the target form, and showed that this yields self-assembly and regeneration after perturbation. Pattern formation is treated as inference about position in a body plan, in contrast to the physical mechanism of reaction-diffusion. The two descriptions address different levels, and their precise relation is unsettled.

## Related concepts
- [[pattern_formation]] — the general study of emergent spatial structure
- [[morphogenesis]] — biological development of form
- [[developmental_systems]] — the developmental context for patterning models
- [[bifurcation_theory]] — the framework for instabilities of homogeneous states
- [[friston_2015_knowing]] — free energy account of pattern regulation

## References
- Turing, A. M. (1952). The chemical basis of morphogenesis. Philosophical Transactions of the Royal Society of London B, 237, 37-72.
- Gierer, A., & Meinhardt, H. (1972). A theory of biological pattern formation. Kybernetik, 12, 30-39.
- Murray, J. D. (2003). Mathematical Biology II: Spatial Models and Biomedical Applications (3rd ed.). Springer.
