---
title: Validation
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/manuscript/06_reproducibility]]"
      - "[[docs/repo_docs/validation]]"
      - "[[docs/tools/mutation_testing]]"
---

# Validation

What the repository checks, from the repository root:

- `python -m pytest -q`: the suite in `code/tests/`, including closed-form oracles
  (`test_math_oracles.py`) and normalization and finiteness audits
  (`test_truth_audit.py`).
- `cognitive-mutation-check`: applies declared mutations of the package's maths and
  requires the suite to fail on each ([[docs/tools/mutation_testing|mutation testing]]).
- `cognitive-validate-docs` and `cognitive-verify-links`: documentation structure
  and wiki links.
- `cognitive-build-manuscript`: rebuilds the manuscript with injected values and
  a build manifest ([[docs/manuscript/06_reproducibility|reproducibility]]).
- `cognitive-benchmark`: small timing runs.

Continuous integration is `.github/workflows/quality.yml`. See also
[[docs/repo_docs/validation|the validation guide]].
