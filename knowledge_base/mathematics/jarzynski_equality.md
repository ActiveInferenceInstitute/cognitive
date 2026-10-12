---
title: Jarzynski Equality
description: The exact nonequilibrium identity relating the exponential average of work to equilibrium free-energy differences, its relation to the second law and fluctuation theorems, and its formal parallels in variational inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - stochastic_thermodynamics
  - free_energy
  - fluctuation_theorems
semantic_relations:
  - type: relates
    links:
      - "[[fluctuation_theorems]]"
      - "[[entropy_production]]"
      - "[[fluctuation_dissipation]]"
      - "[[thermodynamics]]"
      - "[[free_energy_principle]]"
aliases:
  - jarzynski
wikidata: Q864883
---

# Jarzynski Equality

## Overview
The Jarzynski equality is an exact result of nonequilibrium statistical mechanics. A system starts in equilibrium at inverse temperature $\beta$ and is driven by changing a control parameter $\lambda$ from $\lambda_0$ to $\lambda_1$ according to a fixed protocol, however fast. If $W$ is the work done in one realisation, then
$$\big\langle e^{-\beta W}\big\rangle = e^{-\beta\,\Delta F},$$
where the average is over repetitions of the protocol and $\Delta F=F(\lambda_1)-F(\lambda_0)$ is the equilibrium free-energy difference at the same temperature. The final state of the system need not be in equilibrium.

## Consequences
- **Second law.** Jensen's inequality gives $e^{-\beta\langle W\rangle}\le\langle e^{-\beta W}\rangle$, hence $\langle W\rangle\ge\Delta F$. The average dissipated work $\langle W\rangle-\Delta F$ is non-negative, while individual trajectories may have $W<\Delta F$.
- **Free-energy estimation.** Equilibrium free-energy differences can be estimated from repeated nonequilibrium experiments. In practice the average is dominated by rare low-work trajectories, so estimates converge slowly when dissipation is large.
- **Crooks relation.** A more detailed result, $P_F(W)/P_R(-W)=e^{\beta(W-\Delta F)}$, compares work distributions for forward and time-reversed protocols, and implies the Jarzynski equality (see [[fluctuation_theorems]]).

The equality has been tested in single-molecule pulling experiments on biomolecules, and it is a standard part of stochastic thermodynamics.

## Relation to Active Inference
The connection is a formal parallel, not a derivation. The free energy in the [[free_energy_principle]] is variational free energy, an information-theoretic bound on surprise, whereas the free energy in the Jarzynski equality is thermodynamic. They share a mathematical structure: exponentiated averages recover a free energy or log-evidence, and Jensen's inequality turns the identity into a bound. The same structure is used in annealed importance sampling, which has weights resembling Jarzynski work. Work in stochastic thermodynamics relates information processing in agents to dissipation, which gives a physical reading of the cost of inference, but the principle itself does not rely on this identity.

## Related concepts
- [[fluctuation_theorems]] — the family of exact results that includes the Crooks relation
- [[entropy_production]] — the dissipation that the second law bounds
- [[fluctuation_dissipation]] — the near-equilibrium linear-response counterpart
- [[thermodynamics]] — the classical laws that the equality refines
- [[free_energy_principle]] — a different but formally related use of free energy

## References
- Jarzynski, C. (1997). Nonequilibrium equality for free energy differences. Physical Review Letters, 78, 2690-2693.
- Crooks, G. E. (1999). Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences. Physical Review E, 60, 2721-2726.
- Seifert, U. (2012). Stochastic thermodynamics, fluctuation theorems and molecular machines. Reports on Progress in Physics, 75, 126001.
