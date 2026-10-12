---
title: Spectral Theory
description: The study of the spectrum of linear operators, covering eigenvalue decompositions, the spectral theorem for self-adjoint operators, and applications to Markov chains, diffusion operators and dynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - functional_analysis
  - operators
  - linear_algebra
semantic_relations:
  - type: relates
    links:
      - "[[functional_analysis]]"
      - "[[hilbert_spaces]]"
      - "[[linear_algebra]]"
      - "[[fokker_planck]]"
      - "[[stochastic_processes]]"
wikidata: Q2409122
---

# Spectral Theory

## Overview
Spectral theory generalises the eigenvalue theory of matrices to linear operators on infinite-dimensional spaces. The spectrum of a bounded operator $T$ on a complex Banach space is the set $\sigma(T)$ of complex numbers $\lambda$ for which $T-\lambda I$ has no bounded inverse. In finite dimensions this is exactly the set of eigenvalues. In infinite dimensions it can also contain points of continuous spectrum that are not eigenvalues, which matter for operators such as multiplication or differentiation on function spaces.

## Spectral theorems
- **Compact self-adjoint operators.** A compact self-adjoint operator on a Hilbert space has real eigenvalues, which can accumulate only at zero, and the eigenvectors form an orthonormal basis of the closure of its range, giving the expansion $Tx=\sum_k\lambda_k\langle x,e_k\rangle e_k$ ([[hilbert_spaces]]).
- **Bounded self-adjoint operators.** A bounded self-adjoint operator has real spectrum and admits a spectral resolution $T=\int\lambda\,dE(\lambda)$ in terms of a projection-valued measure $E$.
- **Unbounded operators.** Differential operators such as the Hamiltonian $H=-\tfrac12\Delta+V$ are unbounded, and the self-adjoint case requires care with domains. The spectral theorem still applies and underlies the definition of $e^{-tH}$ (Reed & Simon, 1980).

## Applications
- **Markov chains.** For a reversible chain, the transition operator is self-adjoint in the weighted space $L^2(\pi)$. Its second-largest eigenvalue controls convergence to stationarity, so the spectral gap determines the relaxation time and bounds the mixing time up to logarithmic factors (Levin & Peres, 2017). See [[markov_chain_monte_carlo]].
- **Diffusions.** The generator of a reversible diffusion, equivalently the [[fokker_planck]] operator after a similarity transform to a Schrodinger form, has a spectrum whose gap gives the slowest relaxation rate and whose eigenfunctions describe metastable modes.
- **Path operators.** In the Feynman-Kac representation, evolution over a time $t$ is the semigroup $e^{-tH}$. As $t\to\infty$ the ground-state eigenfunction dominates, which connects spectral properties to long-time path-integral behaviour ([[schrodinger_equation]], [[path_integral_theory]]).
- **Linear stability.** The spectrum of the Jacobian at a fixed point determines local stability ([[linear_stability]]).

## Relation to Active Inference
Spectral methods appear in the analysis of the dynamics that the [[free_energy_principle]] describes. Relaxation to a non-equilibrium steady state, the timescales of belief updating, and the decomposition of flows into dissipative and solenoidal parts can all be examined through the eigenvalues of the relevant operators, as in the study of the Fokker-Planck operator. This is analytical background and not a specific claim of Active Inference.

## Related concepts
- [[functional_analysis]] — the setting for operator theory
- [[hilbert_spaces]] — where the spectral theorem is cleanest
- [[linear_algebra]] — the finite-dimensional case
- [[fokker_planck]] — an operator analysed by its spectrum
- [[linear_stability]] — eigenvalue criteria for stability
- [[schrodinger_equation]] — a prototype for the spectral problem

## References
- Reed, M., & Simon, B. (1980). Methods of Modern Mathematical Physics I: Functional Analysis (rev. ed.). Academic Press.
- Kreyszig, E. (1978). Introductory Functional Analysis with Applications. Wiley.
- Levin, D. A., & Peres, Y. (2017). Markov Chains and Mixing Times (2nd ed.). American Mathematical Society.
