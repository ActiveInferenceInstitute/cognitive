---
title: Fluctuation Theorems
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - stochastic_thermodynamics
  - non_equilibrium
  - statistical_physics
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/non_equilibrium_thermodynamics]]"
      - "[[knowledge_base/mathematics/non_equilibrium_steady_state]]"
      - "[[knowledge_base/mathematics/partition_function]]"
      - "[[knowledge_base/mathematics/entropy_production]]"
aliases:
  - crooks
  - crooks_fluctuation_theorem
  - crooks_relation
---

# Fluctuation Theorems

## Overview
Fluctuation theorems are exact relations for the statistics of work, heat, or entropy production in small systems driven away from equilibrium. They hold for arbitrary driving speed, and the second law emerges from them as an inequality on averages. Individual trajectories can violate the second law; the theorems quantify exactly how rarely.

## Definition / Formalism
Consider a system in equilibrium at inverse temperature $\beta$ that is driven by a protocol changing its parameters from $\lambda_0$ to $\lambda_1$, with work $W$ done along a trajectory and equilibrium free-energy difference $\Delta F=-\beta^{-1}\ln(Z_1/Z_0)$.

Jarzynski equality:

$$\big\langle e^{-\beta W}\big\rangle=e^{-\beta\Delta F}.$$

Crooks relation, comparing the forward protocol with its time reverse:

$$\frac{P_F(W)}{P_R(-W)}=e^{\beta(W-\Delta F)}.$$

Jensen's inequality applied to the Jarzynski equality gives $\langle W\rangle\ge\Delta F$.

The simplest exactly checkable case is an instantaneous quench: the state is drawn from the initial Boltzmann distribution $p_i\propto e^{-\beta E_i}$ and the energies change to $E_i'$, so $W_i=E_i'-E_i$. Then $\langle e^{-\beta W}\rangle=\sum_i p_i e^{-\beta(E'_i-E_i)}=Z_1/Z_0$ exactly. Numerically, with $\beta=1.3$ and three levels, both sides equal 0.97515, while $\langle W\rangle=0.2645$ exceeds $\Delta F=0.0194$.

The detailed fluctuation theorem for total entropy production $\Sigma$ reads $P(\Sigma)/P(-\Sigma)=e^{\Sigma}$ (in units of $k_B$), which implies the integral form $\langle e^{-\Sigma}\rangle=1$ and hence $\langle\Sigma\rangle\ge0$.

## Role in Active Inference
The connection is through shared structure, not through the repo. Fluctuation theorems are written in terms of the log-ratio of forward and time-reversed path probabilities, the same kind of quantity that appears in KL-divergence formulations of the free-energy principle and in work on non-equilibrium steady states. Jarzynski's equality also has the form of a free-energy estimator: it recovers $-\ln Z$ ratios from an average of exponentials, which is the same computation as importance-sampling estimates of model evidence. The discrete belief-updating code in `code/tools/src` simulates no thermodynamic work, so no theorem here is exercised by it. Claims that an agent's free energy is literally a thermodynamic free energy are interpretive and should be flagged as such.

## Related concepts
- [[non_equilibrium_thermodynamics]] — the setting of these theorems
- [[non_equilibrium_steady_state]] — where entropy production persists in steady state
- [[partition_function]] — $Z_1/Z_0$ defines $\Delta F$
- [[entropy_production]] — the quantity whose fluctuations are constrained
- [[importance_sampling]] — the estimator view of Jarzynski's equality
- [[langevin_dynamics]] — a standard stochastic model for which the theorems are derived

## References
- Jarzynski, C. (1997). Nonequilibrium equality for free energy differences. *Physical Review Letters* 78, 2690.
- Crooks, G. E. (1999). Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences. *Physical Review E* 60, 2721.
- Seifert, U. (2012). Stochastic thermodynamics, fluctuation theorems and molecular machines. *Reports on Progress in Physics* 75, 126001.
