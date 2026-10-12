---
title: Onsager Reciprocal Relations
description: Symmetry of the kinetic coefficients linking thermodynamic forces and fluxes near equilibrium, following from microscopic time-reversal invariance, and their role in non-equilibrium thermodynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - non_equilibrium_thermodynamics
  - linear_response
  - symmetry
semantic_relations:
  - type: relates
    links:
      - "[[non_equilibrium_thermodynamics]]"
      - "[[entropy_production]]"
      - "[[fluctuation_dissipation]]"
      - "[[detailed_balance]]"
      - "[[thermodynamics]]"
wikidata: Q901483
---

# Onsager Reciprocal Relations

## Overview
The Onsager relations are a symmetry result for systems slightly out of thermodynamic equilibrium. If a set of thermodynamic forces $X_j$ (gradients of temperature, chemical potential and so on) drives a set of fluxes $J_i$, then in the linear regime
$$J_i=\sum_j L_{ij}X_j,\qquad L_{ij}=L_{ji}.$$
Onsager (1931) derived the symmetry of the matrix of kinetic coefficients $L$ from the time-reversal invariance of microscopic dynamics, combined with an assumption that fluctuations decay on average according to the same linear laws as macroscopic perturbations.

## Content and consequences
- **Cross effects.** The relations connect coupled transport processes. A temperature gradient can drive a particle flux (thermal diffusion) and a concentration gradient a heat flux (the Dufour effect), and the corresponding coefficients are equal. Thermoelectric effects such as the Seebeck and Peltier coefficients are related in the same way.
- **Magnetic fields.** When time reversal also flips an external parameter such as a magnetic field $B$, the relation becomes $L_{ij}(B)=L_{ji}(-B)$. If some variables are odd under time reversal, signs enter accordingly.
- **Entropy production.** The entropy production rate is $\sigma=\sum_i J_iX_i=\sum_{ij}L_{ij}X_iX_j\ge 0$, so the symmetric part of $L$ is positive semidefinite. Linear irreversible thermodynamics thereby takes the form of a gradient-like relaxation ([[entropy_production]]).
- **Fluctuations.** The same coefficients appear in the Green-Kubo expressions for transport coefficients and in the fluctuation-dissipation theorem ([[fluctuation_dissipation]]), tying dissipative response to equilibrium fluctuations.

The relations are restricted to the linear regime near equilibrium. Far from equilibrium, no general symmetry of this kind holds.

## Relation to Active Inference
A structural parallel exists in the way flows are decomposed in the free energy literature. A stationary stochastic dynamics $\dot x=f(x)+\omega$ can be written as $f=(\Gamma+Q)\nabla\ln p^*(x)$, with $\Gamma$ a symmetric positive semidefinite diffusion-related part and $Q$ an antisymmetric solenoidal part. The symmetric part plays a role analogous to the Onsager matrix, while the antisymmetric part corresponds to non-dissipative circulation. This is a formal analogy used in the [[free_energy_principle]] and in treatments of [[non_equilibrium_steady_state]], not a derivation of Active Inference from Onsager's theory. Onsager and Machlup also supplied a path-integral action for fluctuating linear systems, relevant to [[path_integral]] formulations.

## Related concepts
- [[non_equilibrium_thermodynamics]] — the theory in which the relations are stated
- [[entropy_production]] — the positive quantity built from the coefficients
- [[fluctuation_dissipation]] — links dissipation to equilibrium fluctuations
- [[detailed_balance]] — the microscopic reversibility behind the symmetry
- [[langevin_dynamics]] — stochastic dynamics consistent with these relations
- [[non_equilibrium_steady_state]] — the regime where symmetric and solenoidal flows both appear

## References
- Onsager, L. (1931). Reciprocal relations in irreversible processes. I. Physical Review, 37, 405-426.
- Onsager, L. (1931). Reciprocal relations in irreversible processes. II. Physical Review, 38, 2265-2279.
- de Groot, S. R., & Mazur, P. (1962). Non-Equilibrium Thermodynamics. North-Holland.
