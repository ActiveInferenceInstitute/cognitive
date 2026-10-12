---
title: Perturbation Theory
description: Approximation methods that solve a problem containing a small parameter by expanding around an exactly solvable case, with examples from differential equations and quantum mechanics and uses near free energy minima.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - approximation
  - asymptotics
  - expansions
semantic_relations:
  - type: relates
    links:
      - "[[differential_equations]]"
      - "[[quantum_mechanics]]"
      - "[[taylor_series]]"
      - "[[laplace_approximation]]"
      - "[[bifurcation_theory]]"
wikidata: Q10886678
---

# Perturbation Theory

## Overview
Perturbation theory is a family of methods for problems that cannot be solved exactly but lie close to one that can. The problem contains a small parameter $\varepsilon$, and the unknown is written as a series in powers of it, $x=x_0+\varepsilon x_1+\varepsilon^2 x_2+\cdots$, where $x_0$ solves the unperturbed problem. Substituting the series and matching powers of $\varepsilon$ gives a sequence of simpler equations solved order by order.

## Regular and singular problems
In a regular perturbation problem the leading-order solution already approximates the true one uniformly, and extra orders refine it. In a singular problem, for example when $\varepsilon$ multiplies the highest derivative of a differential equation, the naive series fails in some region such as a boundary layer. Methods of matched asymptotic expansions, multiple scales and averaging are then used. Such series are often asymptotic rather than convergent: truncating at a sensible order can give good accuracy even though the full series diverges.

## An example from quantum mechanics
For a Hamiltonian $H=H_0+\lambda V$ with known non-degenerate eigenvalues $E_n^{(0)}$ and eigenstates $|n\rangle$, the first-order energy shift is $E_n^{(1)}=\langle n|V|n\rangle$, and the second-order shift is
$$E_n^{(2)}=\sum_{m\neq n}\frac{|\langle m|V|n\rangle|^2}{E_n^{(0)}-E_m^{(0)}}.$$
The same logic applies to dynamical systems, where a small change in a parameter moves fixed points and stability, and where expansions near a bifurcation reduce the dynamics to a low-dimensional normal form.

## Relation to Active Inference
The connection is mainly methodological. Approximations that expand around a stationary point, such as the Laplace approximation to a posterior, are second-order expansions of the log density around its mode (see [[laplace_approximation]]). Analyses of the free energy principle also use expansions of dynamics around a non-equilibrium steady state, and treatments of fast and slow variables rely on small-parameter arguments (see [[timescale_separation]]). These are applications of a general mathematical tool, and perturbation theory is not specific to the framework.

## Related concepts
- [[differential_equations]] — the typical setting for small-parameter expansions
- [[quantum_mechanics]] — the classic home of the energy-shift formulas
- [[taylor_series]] — the expansion that underlies the method
- [[laplace_approximation]] — a second-order expansion around a mode
- [[bifurcation_theory]] — how small parameter changes alter qualitative behaviour
- [[timescale_separation]] — singular perturbation of slow-fast systems

## References
- Bender, C. M., & Orszag, S. A. (1978). Advanced Mathematical Methods for Scientists and Engineers. McGraw-Hill.
- Holmes, M. H. (2013). Introduction to Perturbation Methods (2nd ed.). Springer.
- Griffiths, D. J. (2005). Introduction to Quantum Mechanics (2nd ed.). Pearson Prentice Hall.
