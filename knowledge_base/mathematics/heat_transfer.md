---
title: Heat Transfer
description: The movement of thermal energy by conduction, convection and radiation, with the governing laws, the heat equation and entropy production, and its modest relation to thermodynamic views of living systems.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - thermodynamics
  - transport_phenomena
  - differential_equations
semantic_relations:
  - type: relates
    links:
      - "[[thermodynamics]]"
      - "[[entropy_production]]"
      - "[[non_equilibrium_thermodynamics]]"
      - "[[diffusion_processes]]"
      - "[[differential_equations]]"
wikidata: Q179635
---

# Heat Transfer

## Overview
Heat transfer is the study of thermal energy moving between bodies or regions because of a temperature difference. Thermodynamics fixes the direction, from hot to cold, and heat transfer describes the rate. Three modes are distinguished, and real situations often combine them.

## Modes and laws
- **Conduction** is transfer through a material without bulk motion. Fourier's law gives the heat flux density $\mathbf{q}=-k\nabla T$, with thermal conductivity $k$.
- **Convection** is transfer by the motion of a fluid. For a surface at temperature $T_s$ in a fluid at $T_\infty$, Newton's law of cooling gives a rate $q=hA(T_s-T_\infty)$ with heat-transfer coefficient $h$ and area $A$.
- **Radiation** is transfer by electromagnetic waves, requiring no medium. A body of emissivity $\varepsilon$ emits power $\varepsilon\sigma AT^4$, where $\sigma\approx5.67\times10^{-8}\ \mathrm{W\,m^{-2}K^{-4}}$ is the Stefan-Boltzmann constant and $T$ is absolute temperature.

Combining Fourier's law with energy conservation yields the heat equation
$$\frac{\partial T}{\partial t}=\alpha\nabla^2T,\qquad \alpha=\frac{k}{\rho c_p},$$
where $\alpha$ is the thermal diffusivity, $\rho$ the density and $c_p$ the specific heat. It is a diffusion equation, so its solutions smooth temperature differences, and it is the prototype parabolic partial differential equation.

## Entropy production
Conduction is irreversible. In a medium with local temperature $T$, the rate of entropy production per unit volume is $k\,|\nabla T|^2/T^2$, which is nonnegative and vanishes only when the temperature is uniform. This is an instance of the second law and a standard example in linear nonequilibrium thermodynamics.

## Relation to Active Inference
The link is weak. Organisms exchange heat with their environment and regulate temperature, which makes thermoregulation a standard example of homeostasis and of allostatic control, and nonequilibrium thermodynamics is background to the free energy principle's account of steady states. The principle's free energy is an information-theoretic quantity, not a thermal one, and heat-transfer laws play no part in its derivation. Interoceptive models of temperature regulation are the more concrete connection.

## Related concepts
- [[thermodynamics]] — the laws that determine the direction of heat flow
- [[entropy_production]] — irreversibility accompanying conduction
- [[diffusion_processes]] — mathematically related stochastic and deterministic transport
- [[non_equilibrium_thermodynamics]] — framework for steady flows and dissipation
- [[homeostatic_control_theory]] — regulation of variables such as body temperature

## References
- Carslaw, H. S., & Jaeger, J. C. (1959). Conduction of Heat in Solids (2nd ed.). Oxford University Press.
- Callen, H. B. (1985). Thermodynamics and an Introduction to Thermostatistics (2nd ed.). Wiley.
- Bergman, T. L., Lavine, A. S., Incropera, F. P., & DeWitt, D. P. (2011). Fundamentals of Heat and Mass Transfer (7th ed.). Wiley.
