---
title: Hypotheses
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[validation]]"
      - "[[docs/manuscript/06_reproducibility]]"
---

# Hypotheses

A hypothesis about model behaviour is only useful here if a test can fail it.
The repository's practice, visible in `code/tests/`, is to state the claim as an
assertion against a quantity computed independently of the code under test
(`test_math_oracles.py`), and to register defects as mutations that the suite
must catch (`cognitive-mutation-check`). Numerical results quoted in the
manuscript are injected from the build rather than typed; see
[[docs/manuscript/06_reproducibility|the reproducibility chapter]]. The package
does not provide hypothesis-testing statistics.
