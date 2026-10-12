---
title: Test-Driven Development
description: A software practice in which a failing automated test is written before the code that satisfies it, and how it applies to numerical and probabilistic code with known analytic answers.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - software_engineering
  - verification
  - numerical_methods
semantic_relations:
  - type: relates
    links:
      - "[[path_integral_implementation]]"
      - "[[path_integral_verification]]"
      - "[[numerical_methods]]"
      - "[[numerical_stability]]"
      - "[[computational_methods]]"
---

# Test-Driven Development

## Overview
Test-driven development (TDD) is a software-engineering practice, popularised by Kent Beck, in which each small unit of behaviour is specified by an automated test before the code that implements it is written. It is a discipline for design and verification, not a mathematical theory, and it appears in this vault because numerical code for stochastic and variational methods depends on it.

## The cycle
1. **Red.** Write a test for a behaviour not yet implemented and confirm that it fails.
2. **Green.** Write the simplest code that makes the test pass.
3. **Refactor.** Remove duplication and improve structure while the whole suite stays green.

The cycle is repeated in short iterations. A failing test first confirms that the test can fail at all, and the suite accumulates as a regression safety net.

## Testing numerical and probabilistic code
Scientific code has no single expected output in the usual sense, so tests are built around mathematical invariants.
- **Analytic limits.** A simulated Wiener process should have $\mathrm{Var}(W_t)=t$, and a path-integral estimate of a Gaussian expectation can be compared with its closed form.
- **Conservation and normalisation.** Probability vectors sum to one, and a propagator or transition matrix preserves total mass.
- **Convergence order.** Halving a step size $h$ should reduce the error of an order-$p$ scheme by a factor near $2^{p}$.
- **Statistical tolerance.** Stochastic tests compare sample statistics with expected values within a bound derived from the standard error, with fixed random seeds for reproducibility.
- **Property-based checks.** Invariants such as symmetry, monotonicity or non-negativity are checked over many generated inputs.

Tests use real computations on real inputs rather than substitutes for the code under test, which keeps agreement with the mathematics genuinely checked. TDD does not prove correctness. It samples behaviours, and it cannot catch an error in the specification itself.

## Relation to Active Inference
There is no formal link. By loose analogy, the red step states an expectation and observes a discrepancy, which parallels prediction error, but this is a metaphor and not a modelling claim. The practical relevance is that implementations of belief updating, free energy and path integrals are easy to get subtly wrong, and checks against known solutions are the main defence (see [[path_integral_verification]]).

## Related concepts
- [[path_integral_implementation]] — numerical code whose correctness is checked this way
- [[path_integral_verification]] — verification of path-integral results against known cases
- [[numerical_methods]] — algorithms whose convergence behaviour tests can measure
- [[numerical_stability]] — a failure mode that tests on extreme inputs can expose
- [[computational_methods]] — the broader context of scientific computing

## References
- Beck, K. (2002). Test-Driven Development: By Example. Addison-Wesley.
- Fowler, M. (2018). Refactoring: Improving the Design of Existing Code (2nd ed.). Addison-Wesley.
- Wilson, G., Aruliah, D. A., Brown, C. T., et al. (2014). Best practices for scientific computing. PLoS Biology, 12, e1001745.
