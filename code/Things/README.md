---
title: Things
type: reference
status: stable
updated: 2026-10-10
semantic_relations:
  - type: indexes
    links:
      - "[[code/Things/AGENTS|Things index]]"
      - "[[docs/api/api_reference|API reference]]"
      - "[[docs/agents/agent_docs_readme|Agent documentation]]"
---

# Things

`code/Things/` is installed as the `Things` Python package (see
`[tool.setuptools]` in `pyproject.toml`). Only two of its subfolders contain
code. The rest are documentation: design notes for agents that were described
but never implemented here.

## Runnable agents

| Folder | Import | What it is |
|---|---|---|
| [[code/Things/Simple_POMDP/README]] | `from Things.Simple_POMDP import SimplePOMDP` | Discrete POMDP agent configured from a YAML path or dict; seeded, validated, with state save and load and six plot types. |
| [[code/Things/Continuous_Generic/README]] | `from Things.Continuous_Generic import ContinuousActiveInference, ContinuousVisualizer` | Precision-weighted gradient descent on free energy over generalized coordinates, plus a plotting and GIF helper. |

Both are exercised by the test suite in `code/tests/` (for example
`test_simple_pomdp.py` and `test_pomdp_edge_paths.py`). Their signatures
are listed in [[docs/api/api_reference|the API reference]].

## Documentation-only folders

These folders contain markdown (and, for `Generic_Thing`, saved PNG images) but
no importable Python. Read them as design notes, not as descriptions of an
installed API.

- [[code/Things/Ant_Colony/README|Ant_Colony]]: stigmergic foraging concept.
- [[code/Things/BioFirm/README|BioFirm]]: a firm or bioregion as an active inference agent.
- [[code/Things/Generic_POMDP/README|Generic_POMDP]]: matrix-configured discrete agent design; the working discrete code is `Simple_POMDP` and the `cognitive` package.
- [[code/Things/Generic_Thing/README|Generic_Thing]]: the abstract definition of an agent.
- [[code/Things/Baseball_Game/README|Baseball_Game]]: archived concept study.
- [[code/Things/KG_Multi_Agent/README|KG_Multi_Agent]]: agents on a knowledge graph.
- [[code/Things/Path_Network/README|Path_Network]]: agents on a dynamic network.
- [[code/Things/ActiveInferenceInstitute/README|ActiveInferenceInstitute]]: short description of the Institute.

`code/Things/agents/examples/swarm.md` is a single note, not a package.

## Related

- [[code/Things/AGENTS|Things index]]
- [[knowledge_base/agents/architectures_overview|Agent architectures overview]]
- [[knowledge_base/agents/index|Agent index]]
