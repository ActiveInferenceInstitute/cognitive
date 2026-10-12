---
title: Kubo Formulas
description: Linear response theory results expressing how a system responds to a weak perturbation in terms of equilibrium correlation functions, including the Kubo response formula and the Green-Kubo relations for transport coefficients.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistical_physics
  - linear_response
  - transport
semantic_relations:
  - type: relates
    links:
      - "[[fluctuation_dissipation]]"
      - "[[onsager_relations]]"
      - "[[non_equilibrium_thermodynamics]]"
      - "[[statistical_physics]]"
      - "[[fluctuation_theorems]]"
---

# Kubo Formulas

## Overview
The Kubo formulas, derived by Ryogo Kubo in 1957, give the linear response of a system in thermal equilibrium to a small external force in terms of correlation functions of the unperturbed system. They underlie the fluctuation-dissipation theorem and supply exact expressions for transport coefficients such as conductivity, viscosity and diffusivity.

## Response formula
Let the system have Hamiltonian $H_0$ and be perturbed by $H'(t)=-A\,F(t)$, where $F(t)$ is a weak external field coupled to the observable $A$. To first order in $F$, the change in the average of an observable $B$ is
$$\delta\langle B(t)\rangle=\int_{-\infty}^{t}\chi_{BA}(t-t')\,F(t')\,dt',$$
with the response function
$$\chi_{BA}(t)=\frac{i}{\hbar}\,\theta(t)\,\langle[B(t),A(0)]\rangle_0 .$$
Here $\theta$ is the Heaviside step function, the average is over the equilibrium ensemble of $H_0$, and $B(t)$ evolves under $H_0$. In the classical limit the commutator becomes a Poisson bracket, and the response relates to a correlation function as $\chi_{BA}(t)=-\beta\,\theta(t)\,\tfrac{d}{dt}\langle B(t)A(0)\rangle_0$ with $\beta=1/k_BT$. This is a form of the fluctuation-dissipation theorem: spontaneous fluctuations determine the dissipative response.

## Green-Kubo relations
For transport, the same reasoning gives coefficients as time integrals of equilibrium autocorrelations of fluxes. The self-diffusion coefficient in one Cartesian direction is
$$D=\int_0^\infty\langle v_x(0)\,v_x(t)\rangle\,dt,$$
and the shear viscosity is
$$\eta=\frac{V}{k_BT}\int_0^\infty\langle P_{xy}(0)\,P_{xy}(t)\rangle\,dt,$$
where $P_{xy}$ is an off-diagonal component of the stress tensor. Analogous expressions hold for electrical and thermal conductivity. The results are exact within linear response but require that the correlation functions decay fast enough for the integrals to converge, which can fail in low dimensions or near critical points.

## Role in Active Inference
The free energy principle describes systems at a non-equilibrium steady state with random fluctuations, and some treatments there connect dissipative flows to the form of the stochastic dynamics. The Kubo formulas apply near equilibrium and are an input to that thermodynamic background, not a part of Active Inference itself. The link is therefore contextual: shared mathematics of fluctuations, response and dissipation, rather than a derivation of one from the other.

## Related concepts
- [[fluctuation_dissipation]] — the general relation between fluctuations and response
- [[onsager_relations]] — symmetry of the linear transport coefficients
- [[non_equilibrium_thermodynamics]] — setting for irreversible processes
- [[statistical_physics]] — the equilibrium ensembles used in the averages
- [[fluctuation_theorems]] — results valid beyond linear response

## References
- Kubo, R. (1957). Statistical-mechanical theory of irreversible processes. I. General theory and simple applications to magnetic and conduction problems. Journal of the Physical Society of Japan, 12, 570-586.
- Kubo, R., Toda, M., & Hashitsume, N. (1991). Statistical Physics II: Nonequilibrium Statistical Mechanics (2nd ed.). Springer.
