---
title: Cognitive API reference
type: reference
status: stable
updated: 2026-10-09
---

# Cognitive API reference

The importable package is `cognitive` (source in `code/tools/src`). The agent
families live in `Things` (source in `code/Things`). Every snippet on this page
and in [`api_examples.md`](api_examples.md) was executed against the current
tree.

## Exports of `cognitive`

`cognitive.__all__` lists: `ActiveInferenceDispatcher`, `ActiveInferenceFactory`,
`ActiveInferenceModel`, `AdaptiveControl`, `DiscreteGenerativeModel`,
`HomeostaticControl`, `HomeostaticFactory`, `HomeostaticInference`,
`InferenceConfig`, `InferenceMethod`, `MatrixPlotter`, `ModelState`,
`PolicyAggregation`, `PolicyType`, and the probability helpers
`compute_entropy`, `ensure_matrix_properties`, `expected_free_energy`,
`kl_divergence`, `softmax`. `ControlMode`, `ObservationModel`, `StateSpace`, and
`TransitionModel` are exported from `cognitive.models.active_inference`.

## Discrete model and dispatcher

```python
import numpy as np

from cognitive import (
    ActiveInferenceDispatcher,
    DiscreteGenerativeModel,
    InferenceConfig,
    ModelState,
)

model = DiscreteGenerativeModel(
    A=np.array([[0.9, 0.1], [0.1, 0.9]]),
    B=np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2),
    C=np.array([0.0, 1.0]),
    D=np.array([0.5, 0.5]),
    E=np.array([0.5, 0.5]),
)
config = InferenceConfig("variational", "discrete", 2, 0.5, 1.0, seed=7)
dispatcher = ActiveInferenceDispatcher(config, model)

state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
beliefs = dispatcher.dispatch_belief_update(1, state)
policies = dispatcher.dispatch_policy_inference(state)
assert np.isclose(beliefs.sum(), 1.0) and np.isclose(policies.sum(), 1.0)

total, risk, ambiguity, epistemic = model.expected_free_energy(model.D, 0)
assert np.isclose(total, risk + ambiguity)
```

Matrix conventions: `A[o, s] = P(o | s)`, `B[s_next, s_prev, a]`, `C` holds
log-preferences over observations, `D` and `E` are state and action priors.

`DiscreteGenerativeModel` (frozen dataclass) provides `from_config`,
`predict_states`, `predict_observations`, `posterior`, `expected_free_energy`
(returns total, risk, ambiguity, epistemic gain; the total is risk plus
ambiguity, and the epistemic gain is a diagnostic already contained in
ambiguity), `enumerate_policies(horizon, limit=None)`, and
`evaluate_policy(beliefs, policy, discount_factor, epistemic_weight=0.0)`.

`InferenceConfig` takes `method`, `policy_type`, `temporal_horizon`,
`learning_rate`, `precision_init` and the optional `num_samples`, `temperature`,
`discount_factor`, `exploration_weight`, `policy_limit`, `policy_aggregation`,
`seed`, and `custom_params`. Only `policy_type="discrete"` is implemented; other
values raise `ValueError`, as do out-of-range horizons, learning rates,
precisions, temperatures, sample counts, discount factors, and exploration
weights.

`ActiveInferenceDispatcher(config, model, rng=None)` provides
`dispatch_belief_update(observation, current_state)`,
`dispatch_policy_inference(state, goal_prior=None)`,
`calculate_expected_free_energy`, and `update_precision(prediction_error)`.
`ActiveInferenceFactory.create` and `create_from_yaml` build a dispatcher from
objects or from a YAML mapping with a `generative_model` (or `model`) section;
unknown or missing keys raise `ValueError`.

`ActiveInferenceModel` is the abstract base for stateful models. It supplies
`update_beliefs`, `infer_policies`, `update_precision`, `calculate_free_energy`,
`get_state`, `save_state`, and `load_state`; subclasses implement `step`,
`visualize`, `_load_config`, `_initialize_matrices`, and `_initialize_state`.

## Agent families

- `Things.Simple_POMDP.SimplePOMDP(config)` accepts a YAML path or a dict with
  the sections `model`, `state_space`, `observation_space`, `action_space`,
  `matrices` (`A_matrix` to `E_matrix`), `inference`, and `visualization`, plus
  an optional `seed`. Unknown sections or fields are rejected. Methods: `step`,
  `run`, `reset`, `save_state`, `load_state`, `visualize`. The module also
  exports `SimplePOMDPState` and `compute_expected_free_energy`.
- `Things.Continuous_Generic.ContinuousActiveInference` takes `n_states`,
  `n_obs`, `n_orders`, `dt`, `alpha`, `observation_matrix`, `dynamics_matrix`,
  `observation_precision`, `state_precision`, `precision_learning_rate`,
  `dynamics_weight`, and `seed`. `step(observation)` returns the action and the
  free energy; `reset` restores beliefs. `ContinuousState` holds the beliefs.
- `Things.Continuous_Generic.ContinuousVisualizer(output_dir)` provides
  `plot_belief_evolution`, `plot_generalized_coordinates_relationships`,
  `save_animation` (Pillow GIF), and `create_summary_plot`.

## Homeostatic control

`HomeostaticFactory.create_basic(config)` and `create_adaptive(config)` accept a
YAML path or a dict with `state_spaces` (`environment`, `observation`,
`action`), `observation_model`, `transition_model`, and `inference`, plus the
optional `target_state`, `initial_beliefs`, `action_prior`, and `seed`. Every
action label needs a transition matrix. The result is a `HomeostaticInference`
with `step`, `select_action`, `update_parameters`, `reset`, and `visualize`.

## Utilities

- `cognitive.utils.matrix_utils`: `ensure_matrix_properties`, `compute_entropy`,
  `softmax`, `kl_divergence`, `expected_free_energy` (equal to the first element
  of `DiscreteGenerativeModel.expected_free_energy`).
- `cognitive.models.matrices.matrix_ops`: `MatrixOps`, `MatrixLoader`,
  `MatrixInitializer`, `MatrixVisualizer` (data adapters for plots).
- `cognitive.visualization.matrix_plots`: `MatrixPlotter(output_dir,
  style_config)`, `StateSpacePlotter`, `NetworkPlotter`.
- `cognitive.utils.create_node.NodeCreator(config_path)` renders knowledge-base
  nodes from the templates named in `config.yaml`.
- `cognitive.utils.visualization.NetworkVisualizer` builds and draws the
  knowledge-base link graph.

## Commands

Declared under `[project.scripts]` in `pyproject.toml`:

| Command | Target |
|---|---|
| `cognitive-create-node` | `cognitive.utils.create_node:main` |
| `cognitive-verify-links` | `scripts.verify_links:main` |
| `cognitive-validate-docs` | `scripts.validate_docs:main` |
| `cognitive-normalize-frontmatter` | `scripts.normalize_frontmatter:main` |
| `cognitive-kb-index` | `scripts.kb_index:main` |
| `cognitive-benchmark` | `cognitive.benchmarks:main` |
| `cognitive-build-manuscript` | `scripts.build_manuscript:main` |
| `cognitive-mutation-check` | `scripts.mutation_check:main` |
