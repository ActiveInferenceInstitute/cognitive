---
title: Fluctuation-Dissipation Relations
description: Relations tying a system's spontaneous thermal fluctuations to its linear response and dissipation, including the Einstein relation, and the analogous noise and flow decomposition in free-energy dynamics.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - statistical_physics
  - nonequilibrium
  - stochastic_dynamics
semantic_relations:
  - type: relates
    links:
      - "[[langevin_dynamics]]"
      - "[[fluctuation_theorems]]"
      - "[[entropy_production]]"
      - "[[non_equilibrium_thermodynamics]]"
---

# Fluctuation-Dissipation Relations

## Overview
The fluctuation-dissipation theorem (FDT) states that for a system in thermal equilibrium, the way it responds to a small external perturbation is determined by how it fluctuates spontaneously. The same microscopic interactions that dissipate energy as friction also produce random forces, so noise strength and damping cannot be chosen independently. Callen and Welton (1951) formulated the quantum and classical forms for general linear systems, building on Einstein's and Nyquist's earlier results; Kubo (1966) gave the linear-response derivation.

## Formal statements
- **Einstein relation.** For a Brownian particle, diffusion constant $D$ and mobility $\mu$ satisfy $D=k_BT\,\mu$.
- **Langevin form.** For $\gamma\dot x=-\nabla V(x)+\xi(t)$, equilibrium at temperature $T$ requires white noise with $\langle\xi(t)\xi(t')\rangle=2\gamma k_BT\,\delta(t-t')$. The stationary density is then Boltzmann, $p\propto e^{-V/k_BT}$ ([[langevin_dynamics]]).
- **Linear response (classical).** The response function is $\chi(t)=-\beta\,\frac{d}{dt}C(t)$ for $t>0$, where $C$ is the equilibrium autocorrelation of the observable and $\beta=1/k_BT$.
- **Johnson-Nyquist noise.** Voltage noise across a resistor $R$ has spectral density $4k_BTR$ per unit bandwidth.

The relation holds only at equilibrium. Out of equilibrium, violations quantify distance from detailed balance, and extensions such as the Harada-Sasa equality connect these violations to dissipated heat. Fluctuation theorems (Jarzynski, Crooks) give exact nonlinear counterparts ([[fluctuation_theorems]]).

## Role in Active Inference
The free-energy principle uses a Langevin equation with random fluctuations and decomposes the flow at steady state into $f=(Q-\Gamma)\nabla\ln p^*$, where $\Gamma$ is the diffusion (noise) tensor and $Q$ an antisymmetric matrix. Tying the dissipative part of the flow to the fluctuation amplitude $\Gamma$ is a counterpart of the FDT, and it reduces to the equilibrium relation when $Q=0$. For $Q\neq0$ the system is out of equilibrium and the equilibrium FDT does not hold as stated. The correspondence is thus structural, valid for the stationary-density formalism, not an independent derivation of the equilibrium theorem. Whether biological systems satisfy these conditions is an empirical question and the principle's critics note that the assumptions are strong.

## Related concepts
- [[langevin_dynamics]] — noise and friction dynamics where the relation is applied
- [[diffusion_processes]] — stochastic setting for the diffusion coefficient
- [[fluctuation_theorems]] — exact nonequilibrium generalisations
- [[entropy_production]] — dissipation measure that vanishes at equilibrium
- [[non_equilibrium_steady_state]] — case with non-zero probability current
- [[thermodynamics]] — equilibrium framework in which the FDT originates

## References
- Callen, H. B. and Welton, T. A. (1951). Irreversibility and generalized noise. *Physical Review*, 83, 34-40.
- Kubo, R. (1966). The fluctuation-dissipation theorem. *Reports on Progress in Physics*, 29, 255-284.
