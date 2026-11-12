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
| Test strength | Would the suite notice a wrong line? | `scripts.mutation_check`, retained in the `tests` CI job |
| Coverage | Does branch coverage meet the configured whole-package floor? | `./run.sh test-full` |
| Static | Types and style | `./run.sh lint`, `./run.sh types` |
| Documentation | Frontmatter, links, term policy, catalogs, manuscript | `./run.sh docs` |
| Manuscript | Do figures and variables rebuild and render from the package? | `./run.sh manuscript`, add `--pdf` for PDF |
| Runtime | Do the benchmarks still run? | `cognitive-benchmark --repetitions 1` |
| Packaging | Does the wheel build and do the console scripts start? | Locked build, metadata check and independent wheel smoke step in CI `tests` |

## Order to run before a change

```bash
./run.sh lint
./run.sh types
./run.sh test-full
./run.sh docs
./run.sh gnn
./run.sh manuscript
uv run --locked --extra dev python -m scripts.mutation_check --json
```

The complete development loop and benchmarks are described in
[`docs/development/README.md`](../development/README.md). CI runs it in
`.github/workflows/quality.yml` on Python 3.12.

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
