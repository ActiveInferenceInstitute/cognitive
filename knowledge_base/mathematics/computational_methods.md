---
title: Computational Methods
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - numerics
  - algorithms
  - overview
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/numerical_methods]]"
      - "[[knowledge_base/mathematics/numerical_stability]]"
      - "[[knowledge_base/mathematics/optimization_theory]]"
      - "[[knowledge_base/mathematics/inference_algorithms]]"
---

# Computational Methods

## Overview
Computational methods are the algorithms that turn the mathematics of inference into numbers: normalising distributions, iterating updates to convergence, sampling, and optimising. The mathematics fixes what is computed; these methods determine whether the result is accurate, stable and affordable.

## Definition / Formalism
Four families recur in active inference work.

- **Exact finite computation.** For discrete state spaces, posteriors and expected free energy are sums over states and observations: a matrix-vector product $q(o)=A\,q(s)$ followed by normalisation. Cost is polynomial in the number of states; the policy space grows as $|\mathcal A|^{T}$, which is why `enumerate_policies` in the package takes a `limit`.
- **Fixed-point and gradient iteration.** Variational updates and message passing iterate until a convergence threshold ([[convergence_control]], [[inference_algorithms]]). Stepping along a natural gradient changes the geometry of the iteration, not the fixed point ([[natural_gradients]]).
- **Stochastic approximation.** Monte Carlo, importance sampling and particle filters replace intractable sums or integrals with samples, with error shrinking like $O(N^{-1/2})$ ([[importance_sampling]], [[monte_carlo_methods]]).
- **Optimisation.** Parameter fitting and hyperparameter search ([[optimization_theory]], [[hyperparameter_optimization]]).

Floating-point limits dominate the implementation details. In IEEE double precision $\exp(709)\approx8.2\times10^{307}$ is near the largest representable value (about $1.8\times10^{308}$), so slightly larger arguments overflow, while $\exp(-745)\approx5\times10^{-324}$ is the smallest denormal and $\exp(-746)$ underflows to zero. Subtracting the maximum before exponentiating keeps a softmax finite, and log-sum-exp gives $\ln(e^{1000}+e^{1001})=1001.313$ where the direct computation overflows.

## Role in Active Inference
The package makes these choices explicit, and they are worth knowing when reading results. `MatrixOps.softmax` and `utils.matrix_utils.softmax` subtract the maximum and clip the shifted argument to $[-745,\,709]$. A module-level constant `EPS = 1e-12` guards divisions and logarithms. The two entropy implementations differ slightly: `MatrixOps.compute_entropy` masks zero probabilities exactly, while `matrix_utils.compute_entropy` uses $\ln(p+\text{EPS})$. The `kl_divergence` functions add `EPS` to both arguments, which is harmless for strictly positive distributions but turns an infinite divergence into a finite, large number: $D_{\mathrm{KL}}((0.5,0.5)\,\|\,(1,0))$ evaluates to about 13.1 instead of $\infty$. Callers who need a hard support check should test for it separately.

Runtime is tracked by `cognitive-benchmark`, which times softmax on a 64 by 64 matrix, a dispatcher belief update, a Simple POMDP step and a continuous-inference step, and prints JSON.

## Related concepts
- [[numerical_methods]] — stable softmax, log-sum-exp, normalisation
- [[numerical_stability]] — conditioning and underflow in detail
- [[matrix_operations]] — the linear-algebra kernel of the discrete scheme
- [[scientific_computing]] — reproducibility and tooling around these methods
- [[statistical_computing_implementations]] — worked implementations
- [[statistical_modeling]] — the models these methods fit and evaluate
