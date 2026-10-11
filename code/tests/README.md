---
title: Test suite overview
type: documentation
status: stable
created: 2025-01-01
updated: 2026-10-10
tags:
  - tests
  - testing
  - validation
semantic_relations:
  - type: documents
    links:
      - "[[code/tests/AGENTS|Test suite agent documentation]]"
      - "[[docs/repo_docs/unit_testing|Unit Testing Guide]]"
      - "[[code/tools/src/models/active_inference/AGENTS|Active Inference Models]]"
---

# Test suite overview

The pytest suite in `code/tests/` exercises the `cognitive` package, the two
executable agents under `code/Things/` (`Simple_POMDP`, `Continuous_Generic`),
and the repository tooling in `code/scripts/`. `pyproject.toml` sets
`testpaths = ["code/tests"]`, so a bare `pytest` collects it.

## Running

```bash
python -m pytest -q                                   # whole suite
python -m pytest code/tests/test_matrix_ops.py -v     # one file
python -m pytest -k matrix                            # name filter
python -m pytest -q --cov                             # branch coverage against the floor
```

Do not pass `--cov=<path>`: it narrows what is measured. The `[tool.coverage.*]`
tables in `pyproject.toml` list the measured source trees (`code/tools/src`,
`code/Things`, `code/scripts`), turn on branch coverage, and set the
`fail_under` floor.

Pytest runs with `--strict-markers`; the one registered marker is
`visualization` (tests that generate figures). `conftest.py` selects the
non-interactive `Agg` Matplotlib backend.

## Test files

| File | Covers |
|---|---|
| `test_matrix_ops.py` | `MatrixOps`, `MatrixLoader`, `MatrixInitializer` |
| `test_math_oracles.py` | Closed-form and identity checks of the Active Inference math |
| `test_dispatcher_semantics.py` | Dispatcher parameters that change agent behaviour |
| `test_active_inference_api.py` | Public `cognitive` exports and their behaviour |
| `test_model_lifecycle.py` | Model state lifecycle, persistence, configuration |
| `test_simple_pomdp.py`, `test_pomdp_edge_paths.py`, `test_config_validation.py` | `Things.Simple_POMDP` |
| `visualization/test_continuous_generic.py` | `Things.Continuous_Generic` |
| `test_visualization.py` | `MatrixPlotter`, `StateSpacePlotter`, `NetworkPlotter` |
| `test_boundary_validation.py`, `test_validation_branches.py`, `test_guards_and_cli.py` | Rejected inputs and numerical edge cases |
| `test_verify_links.py`, `test_check_markdown_links.py` | The two link gates in `code/scripts/` |
| `test_validate_docs.py`, `test_normalize_frontmatter.py`, `test_kb_index.py` | Documentation tooling |
| `test_build_pipeline.py` | `build_manuscript.py` and the documentation gates end to end |
| `test_mutation_check.py` | The mutation harness (`mutation_check.py`) |
| `test_truth_audit.py`, `test_version_consistency.py` | Repository-wide truth gates; version agreement |
| `test_tooling_coverage.py` | Plot, graph, and CLI entry points |
| `test_utils.py` | Shared data-generation and logging helpers (not a test module) |

Shared fixtures in `conftest.py` include `output_dir` (session-scoped
temporary directory), `temp_dir`, `sample_matrix_2d`, `sample_matrix_3d`,
`sample_belief_vector`, `sample_matrix_spec`, `sample_markdown_spec`, and
`sample_matrix_data`.

## Continuous integration

`.github/workflows/quality.yml` is the only workflow. On every push and pull
request, four jobs run in parallel: `tests` (the suite with coverage on Python 3.10
and 3.12), `checks` (ruff, mypy, `compileall`, `validate_docs.py`, the manuscript
build and `cognitive-benchmark`, once on 3.12), `mutation-gate`
(`python -m scripts.mutation_check --json`) and `build-package` (build and
install the wheel). Details: [[docs/tools/ci_tools|CI tools]].

## Related

- [[docs/repo_docs/unit_testing|Unit Testing Guide]]
- [[docs/development/contribution_guide|Contribution Guide]]
- [[code/tools/src/models/matrices/AGENTS|Matrix operations]]
- [[code/tools/src/visualization/README|Matrix plotting]]
