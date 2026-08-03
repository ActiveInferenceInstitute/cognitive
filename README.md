---
title: Cognitive Active Inference
type: package
status: stable
---

# Cognitive Active Inference

This repository contains a validated discrete Active Inference package, a
generalized-coordinate continuous agent, matrix utilities, knowledge-base
tools, and reproducible validation commands.

## Install

Use Python 3.10 or newer:

```bash
python -m pip install -e ".[dev]"
```

The editable install exposes the `cognitive`, `Things`, and `scripts` Python
packages and these commands:

```bash
cognitive-create-node --help
cognitive-verify-links . --json
cognitive-validate-docs . --json
cognitive-benchmark --repetitions 10
cognitive-build-manuscript --output build/manuscript
```

## Discrete inference

`DiscreteGenerativeModel` validates the five matrices used by the package:

- `A[o, s] = P(o | s)` and `B[s_next, s_prev, a] = P(s_next | s_prev, a)`;
- `C` contains finite observation log-preferences;
- `D` and `E` are normalized state and action priors.

All dispatcher methods return finite normalized distributions. The dispatcher
supports `variational`, `mean_field`, and `sampling` inference, discrete policy
sequences, explicit risk, ambiguity, epistemic information gain, horizons,
temperatures, and seeds.

```python
import numpy as np

from cognitive import ActiveInferenceDispatcher, DiscreteGenerativeModel, InferenceConfig, ModelState

model = DiscreteGenerativeModel(
    A=np.array([[0.9, 0.1], [0.1, 0.9]]),
    B=np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2),
    C=np.array([0.0, 1.0]),
    D=np.array([0.5, 0.5]),
    E=np.array([0.5, 0.5]),
)
dispatcher = ActiveInferenceDispatcher(
    InferenceConfig(
        method="variational",
        policy_type="discrete",
        temporal_horizon=2,
        learning_rate=0.5,
        precision_init=1.0,
        seed=7,
    ),
    model,
)
state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
beliefs = dispatcher.dispatch_belief_update(1, state)
policies = dispatcher.dispatch_policy_inference(state)
assert np.isclose(beliefs.sum(), 1.0)
assert np.isclose(policies.sum(), 1.0)
```

## Other runtime components

`Things.Simple_POMDP.SimplePOMDP` provides a validated discrete POMDP with
seeded sampling, persistence, histories, expected-free-energy components, and
temporary-directory-friendly plotting. `Things.Continuous_Generic` provides
precision-weighted generalized-coordinate updates and Pillow-backed
multi-frame GIF animation through `ContinuousVisualizer`.

`cognitive.utils.create_node.NodeCreator` resolves configured paths relative to
its YAML file, renders templates safely, and rejects unsafe names. The network
visualizer uses deterministic layouts and handles empty graphs.

## Quality gates

```bash
python -m pytest -q
ruff check .
mypy code/tools/src code/Things code/scripts
python -m compileall -q code
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
python code/scripts/check_markdown_links.py . --json
```

The complete executable manuscript lives in
[`docs/manuscript/`](docs/manuscript/README.md). Its builder generates
deterministic figures, auto-numbered equations and tables, Pandoc citations,
HTML, and a XeLaTeX PDF from `docs/manuscript/config.yaml`.

Tests write artifacts only to temporary directories. Generated reports and
visualization trees are intentionally excluded from version control.

The conceptual material in `knowledge_base/` explains the mathematics and
domain context; executable behavior is defined by the package and its tests.
