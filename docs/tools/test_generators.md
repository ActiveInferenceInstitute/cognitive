---
title: Test generators
type: tool
status: stable
created: 2026-08-02
tags:
  - testing
  - generators
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../../code/tests/README|Test suite]]'
---

# Test generators

Behavior checks are handwritten in `code/tests/` following the repository
conventions: real data and computation, no fixture-only assertions, and
temporary output directories. There is no code-generation tool for tests;
new behavior needs a regression test added by hand (see
`docs/repo_docs/unit_testing.md` and `code/tests/AGENTS.md`).
