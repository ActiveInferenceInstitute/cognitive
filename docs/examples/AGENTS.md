---
title: Examples documentation instructions
type: agents
status: stable
semantic_relations:
  - type: organizes
    links:
      - '[[README|Examples overview]]'
      - '[[../manuscript/README|Executable manuscript]]'
---

# Examples documentation instructions

Examples are documentation of behavior that exists in this repository. Prefer
small examples that can be copied into a Python session and validated against the
public `cognitive` or `Things` packages.

## Sources of truth

- The root `README.md` contains the minimal discrete dispatcher example.
- `docs/manuscript/` contains the end-to-end build, runtime, figure, and
  reproducibility example.
- `code/tests/` contains executable behavior checks.
- `code/Things/` contains self-contained implementations with their own local
  documentation.

Do not present fictional classes, commands, package names, or output paths as
working examples. If an example is theoretical, label it as pseudocode and keep
it out of executable Python fences.

## Validation

```bash
python code/scripts/validate_docs.py --json
python -m pytest -q
```

Examples that write artifacts should accept a caller-selected or temporary output
directory and must not assume an untracked repository output tree.
