---
title: Validation
type: guide
status: stable
created: 2024-02-12
updated: 2026-10-10
tags:
  - validation
  - quality
  - testing
semantic_relations:
  - type: implements
    links:
      - "[[docs/repo_docs/ai_validation_framework|Validation framework]]"
  - type: relates
    links:
      - "[[docs/repo_docs/unit_testing|Unit testing]]"
      - "[[docs/development/README|Development resources]]"
---

# Validation

What "validated" means in this repository, from the code to the documentation.
The documentation checks are detailed in
[[docs/repo_docs/ai_validation_framework|the validation framework]]; this page
lists the whole set and the order to run it.

## Layers

| Layer | Question | Gate |
|---|---|---|
| Mathematics | Does the implementation match a closed form or identity? | `code/tests/test_math_oracles.py`, `test_dispatcher_semantics.py` |
| Inputs | Does a malformed input raise `ValueError` rather than propagate? | `test_boundary_validation.py`, `test_config_validation.py`, `test_validation_branches.py` |
| Behaviour | Do the models keep their lifecycle (step, reset, save, load)? | `test_model_lifecycle.py`, `test_simple_pomdp.py`, `test_pomdp_edge_paths.py` |
| Test strength | Would the suite notice a wrong line? | `python -m scripts.mutation_check` (the `mutation-gate` CI job) |
| Coverage | Is every branch exercised? | `pytest --cov`, floor `fail_under = 90` in `pyproject.toml` |
| Static | Types and style | `ruff check .`, `mypy code/tools/src code/Things code/scripts`, `python -m compileall -q code` |
| Documentation | Frontmatter, links, forbidden terms, catalogs, manuscript | `validate_docs.py`, `verify_links.py`, `check_markdown_links.py` |
| Manuscript | Do figures and variables rebuild from the package? | `python -m scripts.build_manuscript --no-render --output <dir>` |
| Runtime | Do the benchmarks still run? | `cognitive-benchmark --repetitions 1` |
| Packaging | Does the wheel build and do the console scripts start? | `python -m build`, `twine check`, then each `--help` (CI `build-package` job) |

## Order to run before a change

```bash
python -m pytest -q --cov
ruff check .
mypy code/tools/src code/Things code/scripts
python -m compileall -q code
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
python -m scripts.mutation_check
```

The same sequence, plus the manuscript build and benchmarks, is
[`docs/development/README.md`](../development/README.md). CI runs it in
`.github/workflows/quality.yml` on Python 3.10 and 3.12.

## Rules of thumb

- A validator that cannot fail is not a gate. When you add a check, also add the
  test that shows it rejecting a bad input.
- Do not hide a failing step with `|| true` or by redirecting its errors; let the
  step fail or check its exit code.
- A checked-in number in prose or a manuscript must come from a generated
  variable, not a typed literal; `docs/manuscript/` hydrates its values from the
  package when `cognitive-build-manuscript` runs.
- Record a negative result as a result. Do not delete a failing test to make the
  suite green.

## Related documentation

- [[docs/repo_docs/ai_validation_framework|Validation framework]]
- [[docs/repo_docs/unit_testing|Unit testing]]
- [[docs/repo_docs/linking_validation|Link validation]]
- [[docs/manuscript/README|Executable manuscript]]
