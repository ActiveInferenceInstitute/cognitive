---
title: Changelog
type: reference
status: stable
---

# Changelog

This project documents notable changes here. Versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html); entries that change
a computed result are marked **behaviour change** and say what moved and why.

## [Unreleased]

### Behaviour change

Each item names what moved and why. Inputs that were previously accepted and
produced a wrong or arbitrary result now raise `ValueError`.

- **Indices are no longer truncated.** `int(1.9)` is `1`, so
  `matrix_utils.expected_free_energy(..., action=1.9)`,
  `DiscreteGenerativeModel.posterior(1.7, ...)`, `HomeostaticInference.step(1.9)`,
  `SimplePOMDP.step(0.7)`, and the observation arguments of
  `ActiveInferenceModel`/`HomeostaticInference` silently evaluated a different
  action or observation than the caller named. They now raise. Integral values
  of any numeric type (`1`, `np.int64(1)`, `1.0`) are still accepted, and a
  zero-dimensional array is now read as an index rather than as a one-element
  observation vector. The shared check is `matrix_utils.as_integer`.
- **A small posterior is no longer discarded.** `DiscreteGenerativeModel.posterior`
  and `SimplePOMDP._update_beliefs` compared the unnormalized evidence with
  `1e-12`; below it the former returned the prior and the latter a uniform
  belief. Dividing by the evidence is scale-invariant, so likelihoods of `1e-13`
  and `1e-14` now give the exact posterior `(10/11, 1/11)`. Only evidence of
  exactly zero still falls back to the prior (or uniform in `SimplePOMDP`). The
  same `EPS` floor remains in the other normalization helpers (`_normalize` in
  the dispatcher, the homeostatic model and `MatrixOps`, and
  `ensure_matrix_properties`), so inputs whose total mass is below `1e-12` are
  still treated as empty there.
- **`matrix_utils.compute_entropy`** of a deterministic distribution is exactly
  zero. A `log(p + EPS)` floor applied to every entry gave about `-1e-12`; other
  distributions move by about `n * 1e-12` for `n` outcomes.
- **`matrix_utils.ensure_matrix_properties`** rejects empty and non-finite input.
  A NaN column used to fail the `sum > EPS` test and was replaced by a uniform
  distribution, turning corrupt data into a plausible result. A stochastic
  constraint without `non_negative` now rejects negative entries instead of
  returning a "stochastic" matrix that contains them; request `non_negative` to
  clip first.
- **`HomeostaticInference` defaults `exploration_weight` to `0.0`**, matching
  `InferenceConfig`. 1.1.0 moved that default from `0.5` to `0.0` but the
  homeostatic model kept passing `0.5`, so its `infer_policies()` still applied
  half of the extra information-seeking bias that the release removed. Set
  `inference.exploration_weight: 0.5` to keep the old result.
- **`DiscreteGenerativeModel.evaluate_policy`** requires `discount_factor` in
  `(0, 1]`, matching `InferenceConfig`, and a non-empty policy. An empty policy
  used to evaluate to a free energy of exactly zero.
- **`validate_docs`** (and `cognitive-validate-docs`) now also fails when a
  `semantic_relations` link is not a quoted wiki link or a knowledge-base
  catalog is missing or stale. `ensure_matrix_properties` rejects empty and
  non-finite input even when no constraint is requested.
- **`DiscreteGenerativeModel.evaluate_policy`** accepts any iterable of actions
  and raises `ValueError` for a non-numeric `discount_factor`.
- **`NetworkPlotter.plot_message_network_dynamics`** uses a seeded spring layout
  (`seed=0`, settable on the constructor). The layout was previously different
  on every call.

### Added

- `code/scripts/normalize_frontmatter.py` (`cognitive-normalize-frontmatter`)
  keeps every `semantic_relations` link a quoted wiki link. 753 pages stored
  them as nested YAML lists, the residue of unquoted `[[links]]`, so the
  relations were data but invisible to Obsidian's graph and backlinks.
  `validate_docs` now rejects the damaged form.
- `code/scripts/kb_index.py` (`cognitive-kb-index`) generates a `catalog.md` in
  every `knowledge_base/` domain and at its root: each page with its title and
  opening line. `validate_docs` fails when a catalog is missing or stale.
- 158 knowledge-base concept pages for the most-referenced names that had
  no page (learning and plasticity, vision and perception, psychiatry and social
  cognition, statistics, dynamics and control, ecology and swarms, philosophy of
  science), and 72 one-paragraph pages rewritten into articles. Worked examples are
  computed with the package and quote the numbers it returns.

### Fixed

- `StateSpacePlotter.plot_free_energy_landscape` failed with a shape mismatch for
  any non-square array: `np.meshgrid` defaulted to `xy` indexing and transposed
  the grid. It now uses `ij` indexing.
- `ContinuousVisualizer.create_summary_plot` raised `IndexError: tuple index out
  of range` when the history lacked `actions` or `free_energy`. It now names the
  missing key.
- `MatrixPlotter.plot_multi_heatmap`/`plot_belief_evolution` and
  `StateSpacePlotter.plot_belief_evolution` raised a bare
  `IndexError` when given fewer labels than slices or states; they now raise a
  `ValueError` that says how many were needed.
- `verify_links.py` and `check_markdown_links.py` counted a `# comment` inside a
  fenced code block as a heading, so a link to `#install-deps` was accepted
  whenever a shell snippet contained that comment. Fenced blocks are now
  ignored when collecting heading anchors. The repository itself reports no new
  anchor warnings.
- Removed an unreachable duplicate `non_negative` branch in
  `SimplePOMDP._apply_constraint`.

### Documentation

- Restored the `[[link]]` syntax inside markdown and yaml fences that the
  2026-08-02 pass had flattened to plain text. The link checkers already ignore
  fenced code, so the conversion gained nothing and left the linking-standards
  and template pages teaching the wrong syntax.
- `knowledge_base/mathematics/expected_free_energy.md` equated epistemic value
  plus risk with ambiguity plus risk. It now states the canonical decomposition
  used by the package.
- Rewrote `project_structure.md` (it described directories that do not exist)
  and `knowledge_base/quality_assessment.md` (stale counts, subjective ratings),
  cut `docs/DOCUMENTATION_ROADMAP.md` from 809 to 145 lines, and replaced the
  fictional content of the `docs/api/` and `docs/agents/` pages with what the
  package actually exports.

### Tests

- `test_boundary_validation.py` pins each item above with an independently
  derived expectation, and fails against the previous code.
- `test_validation_branches.py` covers the validation and fallback branches of
  the generative model, homeostatic components and configuration, dispatcher
  factory, base-state checks, matrix helpers, node creation, knowledge-network
  visualizer, and plotters.
- Five mutations were added to `mutation_check.py` (22 total), one per defect
  fixed above that a test pins: the posterior threshold, the entropy floor,
  index truncation, non-finite constraint input, and the transposed landscape
  grid. All are killed.

## [1.1.0]

### Mutation gate

`code/scripts/mutation_check.py` and the `cognitive-mutation-check` entry point
apply a declared set of source mutations and require the suite to reject each
one. Coverage says a line ran; this says a wrong line would be noticed. It runs
as its own CI job.

Writing it found four lines that the suite executed but did not check: the
belief blend could apply `learning_rate` to the prior instead of the posterior
(symmetric at the 0.5 every test happened to use), `temperature` could be
dropped from policy inference entirely, the precision clip's upper bound could
move by three orders of magnitude, and the policy aggregation could invert from
a minimum to a maximum. `test_dispatcher_semantics.py` now pins all four.

The harness isolates `PYTHONPYCACHEPREFIX` per run: several mutations replace a
character with another of the same width, which leaves a source file's size and
modification time unchanged within one clock tick, so Python loads the previous
bytecode and the mutation never reaches the tests. That produced an intermittent
false survivor before it was fixed. A mutation that breaks collection rather
than the answer raises instead of counting as a kill.

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
- `policy_aggregation` on `InferenceConfig`, selecting how policies sharing a
  first action are combined. `min` is the existing behaviour and remains the
  default: each first action is scored by its best continuation, an optimistic
  planning assumption. `marginal` is the Bayesian model average of the standard
  formulation, `P(a)` proportional to the sum of `exp(-G / temperature)` over
  policies starting with `a`, so an action reachable by many adequate policies
  is preferred over one reachable by a single excellent policy. The
  implementation returns a soft minimum, which reproduces the marginal exactly
  once passed through the policy softmax.
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
  the whole-package figure was 70%; the measured figure is now 92.6% and the
  floor is 90.
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
