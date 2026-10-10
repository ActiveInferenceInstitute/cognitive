---
title: Unit testing
type: guide
status: stable
created: 2024-02-12
updated: 2026-10-10
tags:
  - testing
  - development
  - quality
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/ai_validation_framework|Validation framework]]"
  - type: relates
    links:
      - "[[docs/implementation/implementation_guides|Implementation guides]]"
      - "[[docs/repo_docs/model_implementation|Model implementation]]"
---

# Unit testing

The suite is pytest, lives in `code/tests/`, and covers the `cognitive`
package, both executable `Things`, and the scripts in `code/scripts/`.

## Running it

```bash
python -m pytest -q                  # whole suite
python -m pytest code/tests/test_simple_pomdp.py -v
python -m pytest -q --cov            # branch coverage against the configured floor
python -m scripts.mutation_check     # does the suite notice a wrong line?
python code/tests/run_tests.py       # wrapper that also writes a JSON report
```

Configuration is in `pytest.ini` and `[tool.pytest.ini_options]` in
`pyproject.toml`: `testpaths = code/tests`, import paths `code/tools` and
`code`, strict markers, and one marker, `visualization`, for tests that draw
figures.

Do not pass `--cov=<path>`. Coverage is configured once, in
`[tool.coverage.*]`, with branch coverage over `code/tools/src`, `code/Things`,
and `code/scripts` and a `fail_under` floor of 90; a `--cov=<path>` argument
silently narrows what is measured.

## Layout

The tests are flat files in `code/tests/`, named for what they cover
(`test_matrix_ops.py`, `test_simple_pomdp.py`, `test_dispatcher_semantics.py`,
`test_validate_docs.py`, `test_verify_links.py`, `test_build_pipeline.py`, and
so on), plus `code/tests/visualization/` for the continuous-agent plots. There
are no `unit/`, `integration/`, or `property/` subfolders, and the `Things`
folders have no tests of their own.

`code/tests/conftest.py` provides the shared fixtures (`output_dir`,
`temp_dir`, `sample_matrix_2d`, `sample_matrix_3d`, `sample_belief_vector`,
`sample_matrix_spec`, `sample_markdown_spec`, `sample_matrix_data`), selects
Matplotlib's `Agg` backend.

## Conventions

- Use real data and real computation. Do not substitute fake objects for the
  code under test.
- Write artifacts only to `tmp_path` or the session `output_dir` fixture.
- Prefer oracles over snapshots: compare against a closed-form quantity or an
  identity (`test_math_oracles.py`) rather than a recorded output.
- Pair a rejected input with the nearest accepted one so a guard that moved by
  one is noticed (`test_validation_branches.py`, `test_boundary_validation.py`).
- When a bug is fixed, add the regression test and, if the line is
  mathematical, a mutation in `code/scripts/mutation_check.py` so the test stays
  load-bearing.
- Test files are `test_*.py`, functions `test_*`.

## Example

A test file written against the public API (this one runs as written):

```python
import numpy as np
import pytest

from cognitive import DiscreteGenerativeModel


@pytest.fixture
def model():
    return DiscreteGenerativeModel(
        A=np.array([[0.9, 0.1], [0.1, 0.9]]),
        B=np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2),
        C=np.array([0.0, 1.0]),
        D=np.array([0.5, 0.5]),
        E=np.array([0.5, 0.5]),
    )


def test_posterior_is_a_distribution(model):
    posterior = model.posterior(0, model.D)
    assert np.isclose(posterior.sum(), 1.0)
    assert posterior[0] > posterior[1]


@pytest.mark.parametrize("horizon", [1, 2])
def test_policy_count(model, horizon):
    assert len(model.enumerate_policies(horizon)) == 2**horizon
```

## Continuous integration

`.github/workflows/quality.yml` runs the suite with branch coverage on Python
3.10 and 3.12, then ruff, mypy, `compileall`, the documentation gates, the
manuscript build, and the benchmarks. A separate job runs
`python -m scripts.mutation_check --json`, and another builds and installs the
wheel. See [[docs/repo_docs/validation|Validation]].

## Related documentation

- [[docs/repo_docs/testing_guidelines|Testing guidelines]]
- [[docs/repo_docs/ai_validation_framework|Validation framework]]
- [[docs/repo_docs/model_implementation|Model implementation]]
- [[docs/tools/mutation_testing|Mutation testing]]
