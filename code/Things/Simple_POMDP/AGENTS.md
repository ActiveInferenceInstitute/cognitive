---
title: Simple POMDP agent
type: agents
status: stable
updated: 2026-10-10
tags:
  - pomdp
  - discrete
  - education
semantic_relations:
  - type: documented_by
    links:
      - "[[code/Things/Simple_POMDP/README]]"
      - "[[code/Things/Simple_POMDP/Simple_POMDP_README]]"
  - type: related_to
    links:
      - "[[docs/api/api_reference|API reference]]"
---

# Simple POMDP agent

Executable. `from Things.Simple_POMDP import SimplePOMDP`; source in
`code/Things/Simple_POMDP/simple_pomdp.py`.

- Depends only on the packages declared in `pyproject.toml` (NumPy, PyYAML,
  Matplotlib).
- Takes a YAML path or dict, validates every section, and raises `ValueError`
  on unknown fields or non-integral indices.
- `step`, `run`, `reset`, `save_state`, `load_state`, `visualize` are the public
  methods; the history it records includes free energy and the expected-free-energy
  components per action.

[[code/Things/Simple_POMDP/Simple_POMDP_README]] has the configuration reference
and a worked example. `Generic_POMDP` is a documentation-only design page, not a
parent class.
