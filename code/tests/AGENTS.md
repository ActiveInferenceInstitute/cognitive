---
title: Test suite agent documentation
type: agents
status: stable
created: 2024-01-01
updated: 2026-10-10
tags:
  - tests
  - testing
  - validation
  - agents
semantic_relations:
  - type: documents
    links:
      - "[[code/tests/README|Test suite overview]]"
      - "[[code/tools/src/models/active_inference/AGENTS|Active Inference Models]]"
      - "[[docs/repo_docs/unit_testing|Unit Testing Guide]]"
---

# Test suite agent documentation

Rules for adding or changing tests in `code/tests/`. The file inventory,
commands, and CI behaviour are in [[code/tests/README|the overview]].

## Rules

- Tests call real code with real arrays and real files. Do not add fixtures
  that fake model behaviour. Write generated artifacts to `tmp_path`, the
  `temp_dir` fixture, or the session `output_dir` fixture, never into the
  repository tree.
- A new public behaviour needs a regression test that fails without the
  change. Prefer an oracle (closed form, identity, rejected input) to a check
  that a call merely returned; see `test_math_oracles.py` and
  `test_boundary_validation.py` for the pattern.
- Tests that create figures may use `@pytest.mark.visualization`
  (`--strict-markers` rejects any unregistered marker) and must inspect the
  produced file or figure.
- Do not pass `--cov=<path>`; run `python -m pytest -q --cov` so the
  configured source trees and the `fail_under` floor in `pyproject.toml`
  apply.
- A mutation declared in `code/scripts/mutation_check.py` must be killed by
  some test. Run `python -m scripts.mutation_check --json` (or
  `--filter <text>` for one mutation) after changing the code it targets.
- `test_utils.py` holds helpers, not tests; `conftest.py` holds shared
  fixtures and the `--mutation` option.

## Before submitting

```bash
python -m pytest -q --cov
ruff check .
mypy code/tools/src code/Things code/scripts
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
```

These are the same gates `.github/workflows/quality.yml` runs.
