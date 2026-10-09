---
title: API documentation
type: documentation
status: stable
updated: 2026-10-09
---

# API documentation

The canonical runtime API is the installed `cognitive` package, plus the
`Things.Simple_POMDP` and `Things.Continuous_Generic` agent packages. Start with
[`api_reference.md`](api_reference.md), then [`api_examples.md`](api_examples.md),
the root [`README.md`](../../README.md), and the executable manuscript in
[`../manuscript/README.md`](../manuscript/README.md). The full page list is in
[`api_documentation_index.md`](api_documentation_index.md).

`code/scripts/validate_docs.py` compiles every Python fence in these pages and
checks that each name in `cognitive.__all__` exists; it does not execute the
fences, so run changed snippets by hand. These pages do not describe classes
that are absent from the packages.
