---
title: Principle of Least Action
description: The variational principle of classical mechanics stating that the physical path makes the action integral stationary, with Hamilton's principle, the Euler-Lagrange equations, and common misconceptions.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - classical_mechanics
  - variational_principles
  - action
semantic_relations:
  - type: relates
    links:
      - "[[lagrangian_mechanics]]"
      - "[[hamiltonian_mechanics]]"
      - "[[euler_lagrange_equations]]"
      - "[[variational_calculus]]"
      - "[[least_action_principle]]"
---

# Principle of Least Action

## Overview
The principle of least action states that the motion of a classical mechanical system between two configurations is the path along which a quantity called the action is stationary. It reformulates Newton's laws as a global statement about whole trajectories and is the foundation of Lagrangian and Hamiltonian mechanics. This page covers the classical principle; its use for stochastic paths and in the free energy principle is treated in [[least_action_principle]].

## Hamilton's principle
For generalised coordinates $q(t)$, the Lagrangian is $L(q,\dot q,t)=T-V$, the kinetic minus the potential energy. The action is
$$S[q]=\int_{t_0}^{t_1}L(q,\dot q,t)\,dt.$$
Among paths with fixed endpoints $q(t_0)$ and $q(t_1)$, the physical path satisfies $\delta S=0$. The calculus of variations then shows that this is equivalent to the Euler-Lagrange equations,
$$\frac{d}{dt}\frac{\partial L}{\partial \dot q}-\frac{\partial L}{\partial q}=0,$$
which reduce to Newton's second law for a particle in a potential.

## Clarifications
- **Stationary, not minimal**: the condition is $\delta S=0$. For short enough time intervals the physical path does give a minimum, but over longer intervals it can be a saddle point, so "least" is a historical name.
- **Variants**: Maupertuis' principle uses the abbreviated action $\int p\,dq$ at fixed energy and yields the shape of the path without its timing. Fermat's principle of least time in optics is an analogous statement.
- **Symmetry**: Noether's theorem links continuous symmetries of the action to conservation laws (see [[noether_theorem]]).
- **Quantum link**: in the path-integral formulation, paths are weighted by $e^{iS/\hbar}$, and the classical path emerges where the phase is stationary as $\hbar\to 0$.
- **Gauge of the Lagrangian**: adding a total time derivative to $L$ does not change the equations of motion.

## Relation to Active Inference
The free energy principle borrows the variational structure of mechanics. In some formulations the time integral of free energy is called free action, and the most likely trajectory of beliefs is the one for which this quantity is stationary, an analogy with Hamilton's principle. This is a formal parallel used in an active research programme, and the quantity is not a mechanical action unless the dynamics are mapped onto a Lagrangian.

## Related concepts
- [[lagrangian_mechanics]] — the formalism built on the action
- [[hamiltonian_mechanics]] — the dual phase-space formulation
- [[euler_lagrange_equations]] — the equations that follow from stationarity
- [[noether_theorem]] — symmetries and conserved quantities
- [[variational_integrators]] — numerical schemes that discretise the action
- [[path_integral]] — the quantum sum over paths weighted by the action

## References
- Goldstein, H., Poole, C., & Safko, J. (2002). Classical Mechanics (3rd ed.). Addison-Wesley.
- Landau, L. D., & Lifshitz, E. M. (1976). Mechanics (3rd ed.). Pergamon Press.
- Gelfand, I. M., & Fomin, S. V. (1963). Calculus of Variations. Prentice-Hall.
