---
title: Changelog
type: reference
status: stable
---

# Changelog

This project documents notable changes here. Versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html); entries that change
a computed result are marked **behaviour change** and say what moved and why.

## [1.1.0]

### Corrected mathematics

**Behaviour change — expected free energy.** The discrete objective now returns
the canonical `risk + ambiguity` rather than `risk + ambiguity - epistemic`.
Ambiguity is `H[q(o)] - I(s; o)`, so it already contains the expected
information gain; subtracting the epistemic term again weighted information
seeking twice and broke the equality between the two standard decompositions of
`G`. `DiscreteGenerativeModel.expected_free_energy` still returns all three
components, and `test_efe_decompositions_agree` pins the identity against
randomly generated models. Policy rankings computed by this package change
wherever the epistemic term differed across actions.

To reproduce the previous objective, set `epistemic_weight=1.0` on
`DiscreteGenerativeModel.evaluate_policy`, or `exploration_weight=1.0` on
`InferenceConfig`.

**Behaviour change — `matrix_utils.expected_free_energy`.** Its ambiguity term
computed `H[q(o)]`, the entropy of the predicted observation, instead of
`E_q(s)[H(P(o | s))]`, the expected entropy of the likelihood. These differ by
the mutual information. The helper and
`DiscreteGenerativeModel.expected_free_energy` now agree exactly for the same
inputs, which `test_matrix_utils_efe_matches_generative_model` verifies. Its
docstring also claimed to subtract an epistemic term it never computed.

**Behaviour change — continuous generalized coordinates.**
`ContinuousActiveInference.step` omitted the `dynamics_matrix` Jacobian
transpose from the dynamic term of the belief update, so the applied update was
not the gradient of the free energy the class reports. The identity dynamics
matrix — the default — hides the error exactly, which is why it survived.
Trajectories under a non-identity `dynamics_matrix` change; trajectories under
the default do not. `test_step_follows_free_energy_gradient` compares the
applied update against a finite-difference gradient for three dynamics
matrices.

**Behaviour change — `matrix_utils.compute_entropy`.** The function normalized
along axis 0 and summed along the requested `axis`. For a vector the two
coincide; for a matrix it returned row entropies computed from
column-normalized values. Normalization now follows the requested axis.

**Behaviour change — homeostatic observation preferences.**
`HomeostaticInference` read the state target directly as an observation
preference, which is only meaningful when the state and observation spaces have
equal size and silently produced a flat preference otherwise. The target is now
pushed through the likelihood, `p*(o) = A @ target`.

**Behaviour change — homeostatic action selection.** A state-shaped control
prior was multiplied into an action-shaped policy behind a shape check, so
whenever the two spaces differed the control mode was dropped without notice.
Each action's predicted state distribution is now scored by the control mode and
reduced to one number per action, and a control mode returning invalid values
raises instead of being ignored.

`_entropy` no longer applies a uniform `log(p + EPS)` floor, so the entropy of a
deterministic distribution is exactly zero rather than slightly negative.

### Added

- `exploration_weight` on `InferenceConfig` and `epistemic_weight` on
  `DiscreteGenerativeModel.evaluate_policy`. The parameter existed and was
  validated but was never read by any code path; it now subtracts an explicit
  additional multiple of the expected information gain. **Its default moved from
  0.5 to 0.0**, which is the canonical objective and matches the behaviour of
  the inert parameter it replaces.
- `dynamics_weight` on `ContinuousActiveInference`, weighting the dynamic term
  consistently in both the reported free energy and the gradient.
- `--no-render` on `cognitive-build-manuscript`, running every stage the package
  owns without Pandoc or XeLaTeX so the pipeline can be exercised where no
  document toolchain exists.
- `docs/policy/documentation_terms.yaml`, the declarative source of the
  documentation term policy, with the files exempt from it named explicitly.
- Test modules `test_math_oracles.py`, `test_model_lifecycle.py`,
  `test_build_pipeline.py`, `test_tooling_coverage.py`,
  `test_config_validation.py`, and `test_version_consistency.py`.
- CI now builds the distribution, installs the wheel, runs every declared
  console script, runs the manuscript pipeline, and runs the benchmarks.

### Changed

- Coverage measures branches and reports every module in the configured source
  tree. The previous configuration reported five modules and showed 93% while
  the whole-package figure was 70%; the floor is now the real number.
- `DiscreteGenerativeModel.enumerate_policies` truncates lazily, so a small
  `limit` over a long horizon no longer materializes the full action product.
- `Things.Simple_POMDP.compute_expected_free_energy` returns the risk term as
  its `pragmatic` component. It previously returned `risk + ambiguity`, which
  is the total, so two of its three numbers were the same quantity.
- The manuscript evaluates its expected-free-energy decomposition at the belief
  reached after the observation sequence. At the uniform prior both transition
  matrices predict the same state distribution, so the published figure was
  identical for every action.
- `docs/manuscript/02_methodology.md` states both decompositions and their
  equality; `03_results.md` and the figure no longer present the epistemic gain
  as a subtracted term.

### Removed

- `code/scripts/normalize_text.py`, which rewrote every occurrence of the
  policy's terms into synonyms across all tracked files, including Python
  sources where a case-insensitive substitution inside an identifier produces a
  syntax error. It converted a gate about accuracy into a gate about
  vocabulary, was wired to nothing, and had no test coverage.
- The tracked `code/tests/test-results.xml`, a stale run report from June 2026.

## [1.0.0]

Initial validated release: discrete generative model and dispatcher, continuous
generalized-coordinate agent, homeostatic control, matrix utilities,
knowledge-base tooling, documentation gates, and the executable manuscript.
