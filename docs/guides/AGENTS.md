---
title: Guides documentation instructions
type: agents
status: stable
semantic_relations:
  - type: organizes
    links:
      - '[[README|Guides index]]'
      - '[[application/README|Application guides]]'
      - '[[learning_paths/README|Learning paths]]'
---

# Guides documentation instructions

The guides directory contains conceptual guides, application notes, and learning
paths. It is not a second implementation package.

## Accuracy boundary

- Runtime examples must use symbols that are importable from the installed
  `cognitive` or `Things` packages.
- A conceptual or research example must be labelled as illustrative and must not
  be presented as an available class, command, or module.
- Use the root [`README.md`](../../README.md) and the executable manuscript as the
  canonical starting points for working code.
- Do not use example package names, community URLs, repository paths, or
  configuration classes that are absent from this repository.

## Validation

Before merging a guide change, run:

```bash
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
```

Python fenced examples must compile under the documentation validator. For a
working API example, add a regression test when practical rather than relying on
an unexecuted code block.
