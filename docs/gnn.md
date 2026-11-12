---
title: Model specification
type: documentation
status: stable
updated: 2026-11-01
tags:
  - gnn
  - generative_model
  - validation
semantic_relations:
  - type: documents
    links:
      - "[[README|Documentation hub]]"
      - "[[../README|Repository README]]"
      - "[[architecture/README|Architecture]]"
      - "[[repo_docs/README|Repository documentation]]"
---

# Model specification

The canonical discrete model is declared in
[Generalized Notation Notation](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation)
(GNN): one plain-text document describes the likelihood, transition tensor,
preferences, priors, and planning horizon. The declaration is the canonical
statement of the model, the configuration in
[`config_examples/discrete_variational.yaml`](../code/tools/src/models/active_inference/config_examples/discrete_variational.yaml)
stays the runtime configuration the factory reads, and
[`test_gnn_conformance.py`](../code/tests/test_gnn_conformance.py) binds the two
together, so editing one side alone fails the suite.

The upstream parser revision is pinned. The declaration is checked with the
revision recorded in [`specs/gnn-pin.json`](../specs/gnn-pin.json) — GNN v5.0.3,
commit `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1` — never with a live checkout,
and the parser files are digest-verified before their verdict is used.

## Files

| Path | Purpose |
| --- | --- |
| [`specs/discrete_variational.gnn.md`](../specs/discrete_variational.gnn.md) | The declaration: `A`, `B`, `C`, `D`, `E`, the variables they connect, the equations, and the ontology annotations. |
| [`specs/gnn-pin.json`](../specs/gnn-pin.json) | The pinned upstream revision — repository, tag, commit — and the SHA-256 digest of every parser file. |
| [`code/tools/src/models/active_inference/gnn_spec.py`](../code/tools/src/models/active_inference/gnn_spec.py) | The reader. Standard library plus NumPy; no upstream import. |
| [`code/tests/test_gnn_conformance.py`](../code/tests/test_gnn_conformance.py) | The binding between the declaration and the runtime model, with negative controls. |
| [`code/tests/gnn_parser_probe.py`](../code/tests/gnn_parser_probe.py) | The only module that imports the upstream parser. Runs as a subprocess under the pinned interpreter and prints JSON. |
| [`code/scripts/validate_gnn_spec.py`](../code/scripts/validate_gnn_spec.py) | The validator: semantics always, pinned parser when a runtime is named. |

## What the declaration is bound to

`ActiveInferenceFactory.create_from_yaml` builds the runtime model from the
canonical configuration, and the conformance test compares:

- `A`, `B`, `C`, `D`, and `E` values and shapes, exactly, against that model;
- the declared dimensions against `num_observations`, `num_states`, and
  `num_actions`;
- `num_hidden_states`, `num_obs`, `num_actions`, `num_timesteps`, and the
  `Time` section's `ModelTimeHorizon` against the model and the canonical
  dispatcher's `temporal_horizon`;
- the declared ontology annotation of every declared variable.

`DiscreteGenerativeModel` reads its `B` array as `B[s_next, s_prev, a]` and
applies `B[:, :, action]`. The declaration holds one 2x2 matrix per action index,
so each declared matrix is exactly the slice the runtime applies: index 0 keeps
the hidden state, index 1 flips it with probability 0.90. The pinned parser also
reports the tensor orientation it sees, and the conformance test requires that
report to say canonical.

The remaining dispatcher keys (`method`, `policy_type`, `learning_rate`,
`precision_init`, `temperature`, `discount_factor`, `exploration_weight`,
`policy_limit`, `policy_aggregation`, `seed`) are inference settings rather than
model parameters; they stay the configuration's business and are not declared
here.

## Adoption context

The patterns and pin discipline this adoption follows are documented upstream in
the [GNN downstream adoption guide](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/integration/downstream_adoption.md)
(main branch; the guide postdates the pinned `v5.0.3` revision, so it is not part
of the pin). This repository uses **pattern D, structure-only declaration**, with
the authored `A`, `B`, `C`, `D`, and `E` declared alongside the structure: the
document states the model, the YAML remains the runtime configuration the factory
loads, no runtime code reads the document, and a conformance test binds the two
sides so editing either alone fails the suite. What that proves: the declared
model is reviewable in one place and cannot drift from the implementation. Two
sibling repositories show the other directions:

- [ActInf_RxInfer](https://github.com/ActiveInferenceInstitute/ActInf_RxInfer) —
  a conjugate Beta-Bernoulli model read by a dependency-free Julia reader;
- [ActiveBlockference](https://github.com/ActiveInferenceInstitute/ActiveBlockference) —
  a discrete grid-world POMDP whose document is generated from the implementation.

The language itself is specified in the
[GNN syntax reference](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/gnn_syntax.md),
and the exemplar corpus under `input/gnn_files/` in the GNN repository is read by
the courses' [GNN syntax track](https://github.com/ActiveInferenceInstitute/courses)
at the same pin.

## Commands

```bash
# Semantics always; the pinned parser when a runtime is named.
python -m scripts.validate_gnn_spec --gnn-runtime <python> --json
GNN_RUNTIME_PYTHON=<python> python -m scripts.validate_gnn_spec
GNN_ROOT=<gnn-checkout> GNN_PYTHON=<python> python -m scripts.validate_gnn_spec

# The binding itself, with or without a runtime.
python -m pytest -q code/tests/test_gnn_conformance.py
```

Provision the pinned runtime with `uv` (the commit is the one in
`specs/gnn-pin.json`):

```bash
uv venv --python 3.12 /tmp/gnn-runtime
uv pip install --python /tmp/gnn-runtime/bin/python \
  "generalized-notation-notation @ git+https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation@8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1"
GNN_RUNTIME_PYTHON=/tmp/gnn-runtime/bin/python python -m scripts.validate_gnn_spec
```

Exit codes: `0` pass, `1` conformance failure, `2` the runtime was unavailable,
so the parser check was not run. Exit `2` is a reported *not run* and never a
pass — the same convention the repository's other gates use.

## What each check proves

- **Semantics** proves the declaration and the canonical configuration agree on
  the numbers, the dimensions, and the horizon. It cannot notice a fault both
  sides share, and it says nothing about the format.
- **Pinned parser** proves the document is readable by a known revision of the
  format: no parse error, no semantic error, a canonical transition-tensor
  orientation, a consistency score of 1.0, and agreement with the reader on
  every variable and ontology annotation. Its files are digest-verified first,
  so a different revision cannot report this verdict.
- **Neither** proves the model is the right model for a task. The declaration
  describes this repository's implementation; it does not certify agreement with
  any published model.

## Changing the model

Change the declaration and the canonical configuration together, then run
`python -m scripts.validate_gnn_spec --gnn-runtime <python>` and
`python -m pytest -q code/tests/test_gnn_conformance.py`. Any figure or example
derived from the canonical configuration changes with it.

## Continuous integration

The `gnn-conformance` job in
[`quality.yml`](../.github/workflows/quality.yml) provisions the pinned revision
in its own Python runtime, runs the validator against it, and re-runs the
conformance tests with `GNN_RUNTIME_PYTHON` set so the parser check cannot be
skipped there. It is a required check: when the runtime cannot be provisioned the
step fails rather than passing quietly.
