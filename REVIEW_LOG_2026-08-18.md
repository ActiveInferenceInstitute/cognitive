---
title: Code and Mathematics Review — 2026-08-18
type: review
status: complete
---

# Code and mathematics review — 2026-08-18

The two preceding repository-wide passes (recorded in
[`REVIEW_LOG_2026-08-02.md`](REVIEW_LOG_2026-08-02.md)) audited documentation.
This pass audited the executable package: the Active Inference mathematics, the
test suite's ability to detect a wrong answer, the coverage and documentation
gates, and the release surface.

## Baseline

Measured before any change, on Python 3.12:

- 102 tests passed; ruff clean; mypy clean.
- `validate_docs.py`: 0 forbidden terms, 0 documentation errors, 0 broken links.
- Coverage reported 92.75% against a 90% floor.

The last figure was the first finding.

## Findings

### 1. Expected free energy double-counted the epistemic drive

`DiscreteGenerativeModel.expected_free_energy` returned
`risk + ambiguity - epistemic`. The two standard decompositions of expected
free energy — `risk + ambiguity`, and `expected surprise - information gain` —
are algebraically identical, because `ambiguity = H[q(o)] - I(s; o)`. The
implementation's own components confirm it: `ambiguity` computed
`E_q(s)[H(A[:, s])]` and `epistemic` computed `I(s; o)` exactly, so subtracting
the second from a total already containing the first weighted information
seeking twice.

Fixed to the canonical total. All three components are still returned. The
deviation was not accidental — `docs/manuscript/02_methodology.md` stated the
subtracting form as its equation — so the manuscript's methodology, results
prose, and figure were corrected with it.

### 2. A second, disagreeing expected-free-energy implementation

`cognitive.utils.matrix_utils.expected_free_energy` computed its ambiguity term
as `H[q(o)]`, the entropy of the predicted observation, rather than the expected
likelihood entropy. Those differ by the mutual information, so the package
shipped two functions of the same name computing different objectives. Its
docstring described subtracting an epistemic term it never computed.

Both now agree exactly, pinned by a cross-implementation test over random models.

### 3. The continuous belief update was not a gradient of its own free energy

`ContinuousActiveInference.step` omitted the `dynamics_matrix` Jacobian
transpose from the dynamic term. Against a finite-difference gradient of
`_compute_free_energy`, the applied update was off by 0.29 in the maximum
component for a non-identity dynamics matrix, and matched exactly for the
identity — which is the default, and the only case the suite exercised.

The update is now exact gradient descent on the reported free energy, verified
by finite difference for three dynamics matrices including the identity. `alpha`
is documented as the step size and the dynamic term's weight moved to an
explicit `dynamics_weight`.

### 4. Two homeostatic paths were correct only by coincidence of dimension

`_preference_vector` read the state target directly as an observation
preference, defined only when the state and observation spaces have equal size
and silently flat otherwise. `select_action` multiplied a state-shaped control
prior into an action-shaped policy behind an `if shapes match` guard, so the
control mode was dropped without notice whenever they differed.

Preferences are now the target pushed through the likelihood. Each action's
predicted state distribution is scored by the control mode and reduced to one
number per action; an invalid control prior raises.

### 5. The suite asserted well-formedness, not correctness

Every existing assertion on expected free energy checked `np.isfinite`. All five
mathematical defects above produce finite, normalized, correctly shaped output,
so the whole suite stayed green through each of them.

Added `test_math_oracles.py`: closed-form entropies and divergences, both EFE
decompositions and their equality, each component against its textbook
definition, the two implementations against each other, degenerate models whose
answers are known analytically (noiseless likelihood, uninformative likelihood,
predictions equal to preferences), a behavioural oracle on action ranking, and
the finite-difference gradient identity.

### 6. The coverage gate measured five of twenty-five modules

`[tool.coverage.report]` carried an `include` list naming five files. The gate
reported 92.75% against a 90% floor; the measured figure across every module in
the configured `source` tree was 70%, with statement-only measurement.

Removed the filter, enabled branch coverage, and set the floor to the real
number. Coverage is now 89.1% (87.7% without a document toolchain, where the
Pandoc render test skips); the floor is 87. Four new test modules cover the
manuscript builder, the documentation validator, the model lifecycle and
persistence paths, the plotting and graph tools, and the configuration guards.

### 7. The documentation gate hid from itself, and a script enforced it by renaming

`validate_docs.py` spelled each of its own forbidden terms as two concatenated
string literals so that its source would not trip the scan it implements.
`code/scripts/normalize_text.py` — uncovered, unwired, and referenced by no
document — walked every tracked file and substituted each term for a synonym,
converting a gate about accuracy into a gate about vocabulary while leaving the
inaccurate content in place. Applied to a Python file, its case-insensitive
substitutions produce syntax errors inside identifiers.

The term list moved to `docs/policy/documentation_terms.yaml` with the files
that must discuss it named as exemptions, so no evasion is needed. The rewriter
was removed. The reasoning is recorded in `docs/policy/README.md`.

### 8. A published figure could not distinguish its own actions

`docs/manuscript/03_results.md` presented a per-action decomposition evaluated
at the uniform prior `D`. Both transition matrices in that experiment are
column-stochastic, so from a uniform prior both predict `q(s) = (0.5, 0.5)` and
every action carried an identical decomposition. The builder now evaluates at
the belief reached after the observation sequence, where the actions separate,
and a test fails if a future configuration flattens it again.

### 9. Release surface

`exploration_weight` was validated by `InferenceConfig` and read by nothing —
an advertised exploration control that did nothing. It now applies an explicit
epistemic weight, with `1.0` reproducing the pre-1.1.0 objective.
`enumerate_policies` iterated the entire action-horizon product before
discarding past `limit`, and now truncates lazily. The package had no `py.typed`
marker, so its annotations were invisible to downstream type checkers. CI gained
narrow token permissions, a concurrency group, dependency caching, and a job
that builds the distribution, installs the wheel, and runs every declared
console script — plus steps running the manuscript pipeline and benchmarks,
neither of which any gate previously executed.

## Verification

Measured after the changes, on Python 3.12:

- 185 tests passed (102 before).
- Branch coverage 89.1% across every configured source module, floor 87.
- ruff clean; mypy clean on 30 source files; `compileall` clean.
- `validate_docs.py`: 0 forbidden terms, 0 documentation errors, 0 broken links,
  0 anchor warnings; manuscript structure valid.
- `verify_links.py`: 20,639 wiki links checked, 0 broken.
- Manuscript builds with and without the document toolchain; the rendered HTML
  contains no unresolved cross-reference or unhydrated variable.
- `python -m build` and `twine check` pass; the wheel carries both `py.typed`
  markers; all five console scripts run from the installed wheel.

## Scope not covered

- Illustrative pseudocode in the ~700 knowledge-base concept pages remains a
  standing editorial activity, as recorded in the previous pass.
- `Things/Simple_POMDP` plotting branches and the `verify_links` CLI paths are
  the largest remaining coverage gaps.
- No mutation testing was run. The oracle tests added here would survive a
  sign flip in several places that a mutation run would find.
