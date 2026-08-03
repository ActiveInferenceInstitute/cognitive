---
title: Style checker
type: tool
status: stable
created: 2026-08-02
tags:
  - style
  - ruff
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[linting_tools|Linting tools]]'
---

# Style checker

Code style is enforced by `ruff format` (project style: ruff only, no
black):

```bash
ruff format code/Things code/tools/src code/scripts code/tests
```

Run `ruff format --check` on the same paths to verify formatting without
writing. See `docs/development/README.md` for the full loop.
