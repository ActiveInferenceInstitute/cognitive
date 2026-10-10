---
title: Simple POMDP
type: reference
status: stable
updated: 2026-10-10
semantic_relations:
  - type: documented_by
    links:
      - "[[code/Things/Simple_POMDP/Simple_POMDP_README]]"
      - "[[code/Things/Simple_POMDP/AGENTS]]"
      - "[[docs/api/api_reference|API reference]]"
---

# Simple POMDP

`SimplePOMDP` (`code/Things/Simple_POMDP/simple_pomdp.py`) is a compact seeded
discrete POMDP agent. Its configuration, given as a YAML path or a dict, has
`model`, `state_space`, `observation_space`, `action_space`, `matrices`
(`A_matrix` to `E_matrix`), `inference`, and `visualization` sections, plus an
optional `seed`. Unknown sections or fields are rejected. `A` and each `B` slice
are column-stochastic, preferences are finite, and `D` and `E` are normalized
distributions.

Methods: `step`, `run`, `reset`, `save_state`, `load_state`, and `visualize`.
`visualize(plot_type)` accepts `belief_evolution`, `free_energy_landscape`,
`policy_evaluation`, `state_transitions`, `observation_likelihood`, and
`efe_components_detailed`; any other value raises `ValueError`. A plot is saved
as `<plot_type>.png` under `visualization.output_dir`. The module also exports
`SimplePOMDPState` and `compute_expected_free_energy`.

Details and a runnable example: [[code/Things/Simple_POMDP/Simple_POMDP_README]].
The tests are in `code/tests/test_simple_pomdp.py`:

```bash
.venv/bin/python -m pytest code/tests/test_simple_pomdp.py -q
```
