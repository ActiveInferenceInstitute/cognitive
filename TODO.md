---
title: Cognitive Active Inference — Project Backlog
type: backlog
status: active
owner: Active Inference Institute
last_reviewed: 2026-08-01
---

# Project Backlog

Repository: cognitive-active-inference (ActiveInferenceInstitute/cognitive)

## Status
- **Owner:** Active Inference Institute
- **Last reviewed:** 2026-08-01 (deepest hostile red-team pass + full implementation)
- **Review scope:** `code/tools/src`, `code/Things`, `code/scripts`, `code/tests`,
  `docs/manuscript`, `knowledge_base`, config and CI. Baseline: 82 tests
  passing; ruff, mypy, compileall, and `validate_docs` all clean. End state:
  **89 tests passing**, every gate green, manuscript (HTML+PDF+figures) and
  benchmarks verified end-to-end.

## Completed / Closed
Every item that was validated in the red-team pass is now implemented in
source/tests/reports. Nothing remains open.

### Findings implemented in the review pass
- **[MEDIUM] Mean-field policy inference double-counted the action/goal prior.**
  The prior entered the softmax twice (baked into EFE *and* applied through
  the mean-field logits), over-concentrating policies. Fixed to apply the prior
  exactly once and added a regression test asserting mean-field matches
  variational under a non-uniform goal prior.
  Affected: `code/tools/src/models/active_inference/dispatcher.py`,
  `code/tests/test_truth_audit.py`.

- **[MINOR] Cross-module private access from the model base class.**
  `ActiveInferenceModel._calculate_expected_free_energy` reached into the
  dispatcher's private `_calculate_expected_free_energy`. The dispatcher method
  is now public (`calculate_expected_free_energy`).
  Affected: `dispatcher.py`, `base.py`.

- **[MINOR] Root-logger side effect in `MatrixPlotter`.** Removed the
  `logging.basicConfig(...)` call from the constructor (global root-logger
  reconfiguration on every instantiation).
  Affected: `code/tools/src/visualization/matrix_plots.py`.

- **[MINOR] Silent identity transition default in homeostatic config.**
  `_load_config` now requires an explicit transition matrix for every action
  label, with a regression test.
  Affected: `code/tools/src/models/active_inference/homeostatic.py`,
  `code/tests/test_active_inference_api.py`.

- **[MINOR] Weak matrix-validation assertion in tests.** Strengthened
  `test_validate_matrix` to assert genuine failures on column-stochasticity and
  non-negativity (previously it only exercised the wrong-shape path).
  Affected: `code/tests/test_matrix_ops.py`.

- **[MINOR] Doc drift in `CLAUDE.md`.** Corrected Python version floor
  (>=3.10) and dependency list to match `pyproject.toml`.

### Majors formerly scoped — now implemented in this pass
- **[MAJOR] `build_manuscript.build()` unconditionally deleted the output dir.**
  An unguarded `shutil.rmtree(output)` could destroy a real directory the
  caller passed via `--output`. Added `_prepare_output`, which only replaces a
  recognized build output (one containing `build_manifest.json` /
  `figure_registry.json`), refuses to delete the working directory, refuses any
  non-empty non-build directory, and recycles empty dirs — with a regression
  test. The manuscript was then rebuilt end-to-end (HTML + PDF + 5 figures +
  manifests) successfully through the guarded path.
  Affected: `code/scripts/build_manuscript.py`,
  `code/tests/test_truth_audit.py`.

- **[MAJOR] Policy-limit truncation silently zeroed unenumerated actions.**
  When `policy_limit < num_actions`, a first action with no enumerated policy
  stayed at `inf` EFE and the softmax silently gave it zero probability purely
  because of a performance cap. `calculate_expected_free_energy` now raises a
  clear error naming the affected action and the offending `policy_limit`; a
  regression test covers the 3-action / limit=2 case.
  Affected: `code/tools/src/models/active_inference/dispatcher.py`,
  `code/tests/test_truth_audit.py`.

### Reproducibility / artifacts verified (not code edits)
- `python code/scripts/build_manuscript.py` produces `build/manuscript/` with
  `manuscript.html`, `manuscript.pdf`, 5 deterministic figures, and the
  `build_manifest.json` / `figure_registry.json` / `manuscript_variables.json`
  provenance files. Verified `combined_sha256` is recorded for auditability.
- `cognitive-benchmarks --repetitions 5` runs all four benchmarks and emits a
  JSON-serializable report.
- Knowledge base (`knowledge_base/`, 675 markdown files, ~23k wiki links):
  strict link audit reports 0 broken explicit links; frontmatter, forbidden
  terms, and python example syntax all validate; 11,112 unresolved concept
  links are intentional by repo convention (concept mode).
