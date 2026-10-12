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

## [2.0.0] - 2026-10-12

### Behaviour change

- **`policy_aggregation` now defaults to `marginal`.** `InferenceConfig` scored each first
  action by the minimum expected free energy over the policies that start with it (`min`), an
  optimistic assumption that judges an action as if the best continuation will be followed.
  The default is now the Bayesian model average of the standard formulation, `P(a)` proportional
  to the sum of `exp(-G / temperature)` over the policies starting with `a`, so an action that
  many adequate policies lead to is preferred over one that a single excellent policy leads to.
  Agents that relied on the old default get different action probabilities; pass
  `policy_aggregation="min"` to keep the previous behaviour. This is the only change made for
  the sake of this decision; the other behaviour changes below accumulated since 1.1.0.

### Added

- **The knowledge base as RDF for the Open Knowledge Network.** `scripts.export_rdf`
  (`cognitive-export-rdf`) writes every knowledge-base page and link as N-Quads and Turtle in the
  graph `https://purl.org/okn/frink/kg/cognitive-kb`, with entity IRIs under
  `https://purl.org/okn/frink/kg/cognitive-kb/`. Pages are `skos:Concept`s (navigation pages are
  `schema:CollectionPage`s) with Dublin Core and schema.org properties; `relates`, `part_of` and
  `organizes` map to `skos:related`, `dcterms:isPartOf` and `dcterms:hasPart`, body links to
  `dcterms:references`, and every other relation name to a term declared a sub-property of
  `skos:semanticRelation`. Provenance is PROV-O, the dataset is described with DCAT and VoID.
  The output is deterministic and uses the link resolver of `verify_links`. `docs/okn/` holds the
  prose description, vocabulary mapping, registry entry, a Turtle sample and generated statistics;
  `validate_docs` fails when the sample or statistics are stale.
- **The long code listings in the knowledge base are trimmed.** 57 pages, led by `calculus`,
  `dynamical_systems`, `free_energy`, `exploration_exploitation` and `epistemic_value`, held about
  44,000 lines of illustrative Python, much of it class listings that the package does not contain
  and several with errors (the `calculus` Jacobian listing was wrong). No page now carries 400 or
  more lines; each keeps a handful of short snippets that were run and checked against closed
  forms or the package. Prose, equations and links are unchanged except where a sentence only
  introduced a removed listing, and headings left with nothing under them were removed.
- `verify_links` walks directories in sorted order. A link whose name matches several pages
  (a repeated file name or alias) resolved to whichever page the filesystem listed first, so the
  same checkout could resolve differently on macOS and Linux; it now resolves identically
  everywhere. The RDF export additionally resolves a repeated name to the alphabetically first path.

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
  other normalizers were brought in line by the scale-invariance item below.
- **Normalization is scale-invariant everywhere.** The dispatcher, homeostatic,
  `MatrixOps`, `ensure_matrix_properties`, `compute_entropy`/`kl_divergence`,
  `SimplePOMDP`, generative-model and base-model normalizers compared the total
  mass with `EPS = 1e-12`, so `[1e-13, 3e-13]` became uniform (or raised
  `must contain positive mass`) while `[1e-11, 3e-11]` became `[0.25, 0.75]`.
  They now share `matrix_utils.normalize_mass` and divide by the sum whenever it
  is positive. Inputs with total mass of at least `1e-12` give identical results
  (the quotient is the same expression); an all-zero slice is handled as before
  (uniform, except `SimplePOMDP`'s column/row helpers, which leave it zero),
  non-finite input is still rejected, and a total that overflows to infinity is
  rescaled by the largest entry instead of yielding zeros.
- **`AdaptiveControl.compute_policy_prior` now depends on `prediction_error`.**
  The exploration term was the same constant for every entry, so the max-shift
  before `exp` cancelled it and the output was identical for any error. The
  tracking score `-|beliefs - goal|` is now scaled by
  `(1 - exploration_weight) / (1 + exploration_weight * prediction_error)`: a
  large error flattens the prior (more exploration), and at an error of zero the
  result equals the old output, which was the same for every error. `prediction_error` must be
  finite and non-negative. With `exploration_weight=0` the error is gated out,
  as before.
- **`SimplePOMDP` reads or rejects every `inference` field.** `inference.seed`,
  `temperature`, and `time_horizon` were accepted and ignored. `seed` now seeds
  the generator (the root `seed` is equivalent; different values for both raise);
  `temperature`, when present, makes `step()` sample the action from
  `softmax(-(G - ln E) / temperature)` (omit it for the previous greedy
  selection), so every configuration that sets `temperature` follows a new
  random stream; `time_horizon`/`temporal_horizon` must be 1; and
  `num_iterations` and `policy_learning_rate`, which nothing read, raise
  `ValueError`. Remove them from existing configurations.
- **`ContinuousActiveInference` bounds learned precision.** With
  `precision_learning_rate > 0` the precision chased `1 / residual**2`: one
  zero-residual step set it to `1e12`, the next gradient step moved the belief by
  `dt * alpha * 1e12`, and the resulting huge residual drove the precision to
  `1e-20` and froze the update. The learned target is now clipped to the new
  `precision_min=1e-3` and `precision_max=1e3`, and the initial
  `observation_precision` must lie inside them when learning is on. Residuals
  between about 0.03 and 30 give the previous result. The sensory step is stable
  only while `dt * alpha * precision_max < 2`.
- **`goal_prior` is applied at the same strength by all three policy methods.**
  Variational and sampling inference folded `-ln(prior)` into the energy, which
  the softmax then divided by the temperature, so the prior entered as
  `ln(prior) / T`; mean-field applied `ln(prior)`. The documented posterior is
  `q(a) proportional to E(a) exp(-G(a) / T)`, which is the mean-field form, so
  `calculate_expected_free_energy(state, goal_prior)` now returns
  `G - T ln(prior)`. Results change for variational and sampling only when
  `temperature != 1`. When `goal_prior` is omitted, mean-field still uses the
  model's `E` and the other two methods still apply no action prior.
- **`SimplePOMDP` applies its `visualization` settings and rejects a per-matrix
  `learning_rate`.** `formats` (a non-empty list from `png`, `pdf`, `svg`) and `dpi`
  were accepted and ignored: every plot was written as a 100-dpi PNG. They now
  decide which files are written and at what resolution, and `style` accepts
  `colormap` and `font_size`. Any other `style` key (`figure_size`,
  `colormap_2d`, `file_format`, ...) now raises `ValueError`, as does a matrix
  `learning_rate`, which nothing read because the matrices are fixed after
  initialization. Remove them from existing configurations.
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

- `code/scripts/markdown_fences.py` is the one place that finds fenced code
  blocks: a line-based tracker that matches the opening fence's character and
  length, treats a shorter run or the other character as content, and runs an
  unterminated fence to the end of the file. `verify_links.py`,
  `check_markdown_links.py`, `kb_index.py`, `validate_docs.py` and
  `build_manuscript.py` use it. `validate_docs` reports unterminated fences in a
  `unterminated_fences` list, which is also a documentation error: a page may not
  leave a fence open. The tracker also finds fences inside blockquotes and callouts
  and on list-item lines, and ends a line only at LF, CR or CRLF.
- `matrix_utils.normalize_mass`, the shared scale-invariant normalizer, and the
  `precision_min`/`precision_max` arguments of `ContinuousActiveInference`.
- `code/scripts/normalize_frontmatter.py` (`cognitive-normalize-frontmatter`)
  keeps every `semantic_relations` link a quoted wiki link. 753 pages stored
  them as nested YAML lists, the residue of unquoted `[[links]]`, so the
  relations were data but invisible to Obsidian's graph and backlinks.
  `validate_docs` now rejects the damaged form.
- `verify_links.py` resolves a wiki link through the `aliases:` of a page's
  frontmatter, as Obsidian does, so `[[plasticity]]` no longer counts as an
  unwritten concept when a page lists it.
- Optional dependency groups: `network` (`scipy`, which networkx needs to lay out
  graphs of 500 or more nodes) and `maintenance` (`tqdm`, imported by the helper
  scripts in `docs/repo_docs/repo_scripts`). `dev` includes `scipy`.
- `code/scripts/kb_index.py` (`cognitive-kb-index`) generates a `catalog.md` in
  every `knowledge_base/` domain and at its root: each page with its title and
  opening line. `validate_docs` fails when a catalog is missing or stale.
- 358 knowledge-base concept pages for the most-referenced names that had no
  page (learning and plasticity, vision and perception, psychiatry and social
  cognition, statistics, dynamics and control, ecology and swarms, philosophy of
  science), and 72 one-paragraph pages rewritten into articles. Worked examples
  are computed with the package and quote the numbers it returns. The second
  batch of 200 was chosen by triaging the 510 link targets with three or more
  inbound links into write, alias and unlink; the references on all of these
  pages were checked against the published records (see Documentation).
- A `description:` line on 55 pages that opened with a formula or a list, so the
  generated catalogs describe them.
- `aliases:` on 124 existing pages so that a link to a synonym or variant name
  (`[[plasticity]]`, `[[executive_function]]`) opens the page that covers it.

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
  ignored when collecting heading anchors, and the non-greedy fence pattern that
  first fixed this was replaced by the line-based tracker: it still read a
  comment inside a four-backtick fence that holds a three-backtick block, or
  after a fence that is never closed, as a heading, and it scanned the contents
  of nested blocks for links. `kb_index.py` toggled "in fence" on every fence
  line and `validate_docs.py` compiled `python` fences found by a pattern; both
  now use the tracker (catalog text is unchanged). `validate_docs` still
  compiles only unindented, closed `python` fences, as before.
- Silent failures in `code/scripts`: `verify_links.py` and
  `check_markdown_links.py` skipped a Markdown file they could not read (the
  first printed and exited 0, the second said nothing); both now list it under
  `unreadable_files` and exit 1, and `validate_docs` reports it. `validate_docs`
  skipped a tracked text file that was not valid UTF-8, exempting it from every
  check, and now reports it. `normalize_frontmatter.py --check` skipped such a
  file and exited 0; it now exits 1. `unquoted_links` read `links: 0`, `""` and
  `false` as an empty list (`or []`); only an absent or null `links` means none.
  `build_manuscript` wrote an empty `preamble.tex` when `preamble.md` had no
  `latex` fence and rendered a PDF without its macros; it now raises.
- `verify_links.py` reported an anchor warning for every `[[page#heading]]` link that
  resolved by file name or alias unless it ran from the repository root, because the
  index held paths relative to the root.
- The normalizers emitted a `RuntimeWarning` ("overflow encountered in reduce") when
  the total mass overflowed, as for entries near `1e308`. The result was right, but a
  warnings-as-errors run failed. The positive-mass checks no longer sum.
- `build_manuscript` wrote a preamble with mixed `\r\n` and `\n` line endings for a
  CRLF source; it now uses `\n` throughout.
- Removed an unreachable duplicate `non_negative` branch in
  `SimplePOMDP._apply_constraint`.

### Documentation

- The knowledge base gained 1,348 concept pages and 72 one-paragraph pages were rewritten
  into articles. The latest 990 cover names that existing pages already linked: named brain
  regions, cortical layers and microcircuits, clinical conditions and cognitive effects;
  perception, memory, attention, motor control, language and social cognition; probability,
  information theory, dynamics, control, optimisation, machine learning, textbook algorithms
  and category theory; philosophy of mind, ethics, logic, phenomenology and the philosophy of
  the special sciences; and ecology, colony biology, cell signalling and research methods.
  Worked examples are computed with the package and quote the numbers it returns. Every
  reference was checked against PubMed, Crossref and publisher records, and the checks
  corrected titles, authors, page ranges, attributions, current Stanford Encyclopedia of
  Philosophy authors, missing hypotheses in theorem statements, a sign error in the Onsager
  flow decomposition, a false claim about non-Lipschitz activations, and a number of
  over-stated claims. A pass over the classic works the vault cites (Shannon, Pearl, Koller
  and Friedman, Broadbent, Baddeley and Hitch, Rescorla and Wagner, Watson, Ryle, Miller,
  Tolman, Hebb, Jaynes, Friston 2010) found one wrong entry, a Helmholtz citation.
- About 165 existing pages that cited nothing now carry a verified `## References` list
  (about 850 entries, each checked against Crossref, PubMed, arXiv, publisher or
  encyclopedia records). Reviewing those pages found and fixed real errors:
  - expected free energy decompositions with a wrong sign or a stray term
    (`active_inference`, `expected_free_energy`, `measure_theoretic_active_inference`,
    `active_inference_connections`), and a rewritten derivation on
    `free_energy_principle/mathematics/expected_free_energy`;
  - a double-counted entropy term in `variational_free_energy`, an inverted
    likelihood ratio in `psychophysics`, an impossible stability condition
    (`|lambda| < 0`) in `advanced_taylor_series`, and wrong signs, factors and drift
    corrections in `fokker_planck` and `langevin_dynamics`;
  - `policy_selection` summed variational free energy where expected free energy
    belongs and named an inverse temperature `temperature`;
  - misattributed works and arguments (a novel cited as a metaverse reference, a critique
    credited to the wrong authors, the "mob psychology" charge, wrong years and
    journals), and unsupported claims (invented patent and licensing tiers on a
    BioFirm page) that were corrected, softened or removed.
  Pages that had both an older `Key References` list and a verified list now have one.
- A second pass looked up all 117 references on 31 philosophy and logic pages against live
  records; one edition year was wrong (Canguilhem 1989, not 1991) and three body sentences were
  tightened.
- The 216 knowledge-base pages that carry 40 or more lines of Python now say, under their
  title, that the snippets are illustrative and are not part of the package unless they import
  it, and point to the API reference for the tested implementations.
- Link lists now carry accurate glosses (4,987 rewritten). About 6,000 links to generic
  labels that no page should exist for (`new_methods`, `version_control`,
  `adaptive_strategies`) were unlinked, and existing pages gained `aliases:` so a
  synonym or variant name opens the page that covers it. About 355 `[[concept]]`
  names remain without a page: each is a specific, well-defined concept (mostly advanced
  mathematics, physics and regional philosophical traditions) kept as a marker for a
  page worth writing.
- `code/Things/ActiveInferenceInstitute/active_inference_institute.md` is a sourced summary of
  the Institute's history, governance, programs, activities and licensing, taken from the
  Institute's public website repository (content through August 2026), with links to the
  authoritative pages.
- `knowledge_base/mathematics/expected_free_energy.md` states the canonical
  decomposition the package uses. It previously equated epistemic value plus risk
  with ambiguity plus risk.
- Four pages (`dynamical_systems`, `expected_free_energy`, `factor_graphs`,
  `free_energy_principle`) closed a code fence with a fence that carried an info
  string, so CommonMark rendered the rest of each page as code. They are fixed, and
  `validate_docs` now fails on an unterminated fence.
- `project_structure.md` describes only directories that exist. The `docs/api/` and
  `docs/agents/` pages describe what the package exports. Generated filler was
  removed from `docs/`: invented RxInfer API names and results, a PyTorch
  integration, learner-analytics scaffolding, and the fictional "Active
  Implementations" in `code/Things/README.md`. The RxInfer notes are rewritten from
  the official documentation.
- `docs/guides/application/parr_2022_chapter_6.md` is a short study guide that
  restates the chapter's four-step workflow in our own words and links to the pages
  that explain each step; it replaces notes that followed the book chapter closely.
  `docs/templates/learning_path_template.md` is a 90-line skeleton instead of 1,373
  lines, and `knowledge_base/swarm_intelligence_implementation.md` states that its
  classes are illustrative sketches.
- Removed the dated review logs, the knowledge-base quality report, the
  documentation roadmap, the stream notes, and the separate test runner
  (`code/tests/run_tests.py`); `pytest` is the one way to run the suite.
- `README.md` is rewritten around diagrams and deep links into the knowledge base.

### Removed

- The vendored RxInfer documentation snapshot (`docs/implementation/rxinfer/docs/`, 56
  files from early 2025). Upstream is at 5.x, so the copy lagged by two major versions;
  the local notes now link to <https://docs.rxinfer.com/stable/> instead. The link
  checker no longer special-cases that directory.
- `.cursorrules` and `docs/tools/cursor_integration.md`: Cursor-specific guidance that
  `AGENTS.md` and `CLAUDE.md` already carry. The `.aii/config.yaml` description now
  describes the repository.
- The `--mutation` pytest option and `test_declared_mutations_all_die`, which ran only
  under that option and so was skipped on every run; the `mutation-gate` CI job runs
  the same check.
- `scripts.verify_links.verify_links()`, a thin wrapper that returned only the
  broken-link list. Use `verify_link_report(root).broken_links`.

### Tests

- `test_boundary_validation.py`, `test_behaviour_changes.py`,
  `test_scale_invariant_normalization.py`, `test_markdown_fences.py` and
  `test_validation_branches.py` pin each behaviour change and fix above with an
  independently derived expectation.
- `mutation_check.py` declares 36 mutations (the normalizer floor, the `goal_prior`
  temperature scale, the `AdaptiveControl` error scale, the precision clip, the
  `SimplePOMDP` settings, the posterior threshold, the entropy floor, index
  truncation, non-finite constraint input, the landscape grid, two fence-tracker
  rules, the falsy-`links` default, and five in the two link checkers: a missing
  file, an explicit missing wiki file, an external link, and the two anchor
  checks). All are killed.

### CI

- The workflow runs the suite on Python 3.10 and 3.12 in one job and every
  version-independent check (ruff, mypy, documentation gate, manuscript build,
  benchmarks) once in another. The document-toolchain step, which asked apt for
  `pandoc-crossref` (Ubuntu does not package it) and so stopped every run before
  pytest from 2026-08-19, now installs a pinned Pandoc and pandoc-crossref pair
  from their release pages.
- The 3.10 leg needs the `tomli` backport (declared in the `dev` extra) because
  `tomllib` is new in 3.11.

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
