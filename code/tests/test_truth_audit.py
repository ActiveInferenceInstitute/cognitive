"""Regression coverage for the repository-wide truth and functionality gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from cognitive import (
    ActiveInferenceDispatcher,
    ActiveInferenceModel,
    DiscreteGenerativeModel,
    InferenceConfig,
    ModelState,
)
from cognitive.benchmarks import run_benchmarks
from cognitive.models.active_inference.dispatcher import ActiveInferenceFactory
from cognitive.models.matrices.matrix_ops import MatrixInitializer, MatrixOps
from cognitive.utils import (
    compute_entropy,
    ensure_matrix_properties,
    expected_free_energy,
    kl_divergence,
    softmax,
)
from cognitive.utils.create_node import NodeCreator
from cognitive.utils.visualization.network_viz import NetworkVisualizer
from PIL import Image
from scripts.verify_links import verify_link_report
from Things.Continuous_Generic import ContinuousActiveInference, ContinuousVisualizer
from Things.Simple_POMDP import SimplePOMDP


def discrete_model() -> DiscreteGenerativeModel:
    return DiscreteGenerativeModel(
        A=np.array([[0.9, 0.1], [0.1, 0.9]]),
        B=np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2),
        C=np.array([0.0, 1.0]),
        D=np.array([0.5, 0.5]),
        E=np.array([0.5, 0.5]),
    )


def test_all_dispatcher_methods_return_valid_distributions() -> None:
    model = discrete_model()
    state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
    for method in ("variational", "mean_field", "sampling"):
        dispatcher = ActiveInferenceDispatcher(
            InferenceConfig(method, "discrete", 2, 0.5, 1.0, num_samples=32, seed=9), model
        )
        beliefs = dispatcher.dispatch_belief_update(1, state)
        policies = dispatcher.dispatch_policy_inference(state)
        assert np.all(beliefs >= 0) and np.isclose(beliefs.sum(), 1.0)
        assert np.all(policies >= 0) and np.isclose(policies.sum(), 1.0)


def test_inference_configuration_rejects_invalid_values() -> None:
    common = {
        "policy_type": "discrete",
        "temporal_horizon": 1,
        "learning_rate": 0.5,
        "precision_init": 1.0,
    }
    invalid = [
        {"policy_type": "continuous"},
        {"temporal_horizon": 0},
        {"learning_rate": 0.0},
        {"precision_init": 0.0},
        {"num_samples": 1},
        {"temperature": 0.0},
        {"discount_factor": 0.0},
        {"exploration_weight": 2.0},
        {"policy_limit": 0},
    ]
    for change in invalid:
        values = {"method": "variational", **common, **change}
        with pytest.raises((ValueError, TypeError)):
            InferenceConfig(**values)


def test_sampling_seed_is_reproducible() -> None:
    model = discrete_model()
    state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
    configs = [InferenceConfig("sampling", "discrete", 1, 0.5, 1.0, seed=22) for _ in range(2)]
    results = [
        ActiveInferenceDispatcher(config, model).dispatch_belief_update(1, state)
        for config in configs
    ]
    assert np.array_equal(results[0], results[1])
    dispatcher = ActiveInferenceDispatcher(
        InferenceConfig("variational", "discrete", 1, 0.5, 1.0), model
    )
    with pytest.raises(ValueError):
        dispatcher.update_precision(-1)
    with pytest.raises(ValueError):
        dispatcher.dispatch_policy_inference(state, np.array([1.0]))


def test_invalid_distributions_and_one_state_initializer_are_rejected_or_valid() -> None:
    with pytest.raises(ValueError):
        MatrixOps.compute_entropy(np.array([2.0, 2.0]))
    with pytest.raises(ValueError):
        MatrixOps.compute_kl_divergence(np.array([1.0, 0.0]), np.array([2.0, 0.0]))
    assert np.array_equal(MatrixInitializer.identity_based((1, 1)), np.ones((1, 1)))
    assert np.isclose(compute_entropy(np.array([0.25, 0.75])), 0.5623351446)
    assert np.isclose(kl_divergence(np.array([0.5, 0.5]), np.array([0.5, 0.5])), 0.0)
    assert np.isfinite(
        expected_free_energy(
            discrete_model().A,
            discrete_model().B,
            discrete_model().C,
            discrete_model().D,
            0,
        )
    )
    assert np.allclose(softmax(np.zeros((2, 3)), axis=1).sum(axis=1), 1.0)
    assert np.allclose(
        ensure_matrix_properties(np.ones((2, 2)), "column_stochastic").sum(axis=0), 1.0
    )
    with pytest.raises(ValueError):
        ensure_matrix_properties(np.ones((2, 2)), "unsupported")
    with pytest.raises(ValueError):
        ensure_matrix_properties(np.ones((2, 2)), ["column_stochastic", "row_stochastic"])
    with pytest.raises(ValueError):
        softmax(np.zeros(2), temperature=0)
    with pytest.raises(ValueError):
        softmax(np.array([np.nan, 1.0]))
    with pytest.raises(ValueError):
        compute_entropy(np.zeros(2))
    with pytest.raises(ValueError):
        kl_divergence(np.ones(2), np.ones(3))
    with pytest.raises(ValueError):
        expected_free_energy(np.eye(2), np.stack([np.eye(2)], axis=2), np.zeros(2), np.ones(2), 9)


def _simple_config(output_dir: Path) -> dict[str, object]:
    return {
        "model": {"name": "one_state", "version": "1"},
        "state_space": {"num_states": 1, "state_labels": ["only"], "initial_state": 0},
        "observation_space": {"num_observations": 1, "observation_labels": ["only"]},
        "action_space": {"num_actions": 1, "action_labels": ["stay"]},
        "matrices": {
            "A_matrix": {
                "shape": [1, 1],
                "initialization": "uniform",
                "constraints": ["column_stochastic"],
            },
            "B_matrix": {
                "shape": [1, 1, 1],
                "initialization": "identity_based",
                "constraints": ["column_stochastic"],
            },
            "C_matrix": {
                "shape": [1],
                "initialization": "log_preferences",
                "initialization_params": {"preferences": [0.0]},
            },
            "D_matrix": {"shape": [1], "initialization": "uniform"},
            "E_matrix": {"shape": [1], "initialization": "uniform"},
        },
        "inference": {"learning_rate": 1.0, "temperature": 1.0},
        "visualization": {"output_dir": str(output_dir)},
        "seed": 4,
    }


def test_simple_pomdp_one_state_and_state_round_trip(tmp_path: Path) -> None:
    model = SimplePOMDP(_simple_config(tmp_path / "plots"))
    model.step(action=0)
    saved = model.save_state(tmp_path / "state.yaml")
    restored = SimplePOMDP(_simple_config(tmp_path / "other-plots"))
    restored.load_state(saved)
    assert restored.state.time_step == model.state.time_step
    assert np.array_equal(restored.state.beliefs, model.state.beliefs)
    invalid = _simple_config(tmp_path / "invalid")
    invalid["unexpected"] = True
    with pytest.raises(ValueError, match="Unknown configuration"):
        SimplePOMDP(invalid)


def test_generative_model_prediction_policy_and_information_paths() -> None:
    model = discrete_model()
    assert np.isclose(model.predict_states(model.D, 0).sum(), 1.0)
    assert np.isclose(model.predict_observations(model.D).sum(), 1.0)
    assert np.isclose(model.posterior(1, model.D).sum(), 1.0)
    assert np.isclose(model.posterior(np.array([0.2, 0.8]), model.D).sum(), 1.0)
    values = model.expected_free_energy(model.D, 0)
    assert all(np.isfinite(value) for value in values)
    policies = model.enumerate_policies(2)
    assert len(policies) == 4
    assert np.isfinite(model.evaluate_policy(model.D, policies[0], 0.95))
    with pytest.raises(ValueError):
        model.predict_states(model.D, 4)
    with pytest.raises(ValueError):
        model.enumerate_policies(0)
    with pytest.raises(ValueError):
        DiscreteGenerativeModel.from_config({"A": model.A})
    with pytest.raises(ValueError):
        model.enumerate_policies(1, limit=0)
    with pytest.raises(ValueError):
        model.predict_states(model.D, 0.5)
    with pytest.raises(ValueError):
        model.posterior(99, model.D)
    with pytest.raises(ValueError):
        model.posterior(np.array([1.0, 0.0, 0.0]), model.D)
    identity = DiscreteGenerativeModel(
        A=np.eye(2),
        B=np.stack([np.eye(2)], axis=2),
        C=np.zeros(2),
        D=np.array([0.5, 0.5]),
        E=np.array([1.0]),
    )
    assert np.array_equal(identity.posterior(0, np.array([0.0, 1.0])), np.array([0.0, 1.0]))
    invalid_matrices = [
        {"A": np.ones(2), "B": model.B, "C": model.C, "D": model.D, "E": model.E},
        {"A": model.A, "B": np.ones((2, 2)), "C": model.C, "D": model.D, "E": model.E},
        {"A": model.A, "B": np.ones((2, 3, 2)), "C": model.C, "D": model.D, "E": model.E},
        {"A": model.A, "B": model.B, "C": np.zeros(3), "D": model.D, "E": model.E},
        {"A": model.A, "B": model.B, "C": model.C, "D": np.zeros(2), "E": model.E},
        {"A": model.A, "B": model.B, "C": model.C, "D": model.D, "E": np.zeros(2)},
    ]
    for values in invalid_matrices:
        with pytest.raises(ValueError):
            DiscreteGenerativeModel(**values)


def test_dispatcher_factory_and_precision_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "dispatcher.yaml"
    config = {
        "method": "sampling",
        "policy_type": "discrete",
        "temporal_horizon": 1,
        "learning_rate": 0.5,
        "precision_init": 1.0,
        "seed": 2,
        "generative_model": {
            "A": discrete_model().A.tolist(),
            "B": discrete_model().B.tolist(),
            "C": discrete_model().C.tolist(),
            "D": discrete_model().D.tolist(),
            "E": discrete_model().E.tolist(),
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    dispatcher = ActiveInferenceFactory.create_from_yaml(config_path)
    assert dispatcher.update_precision(0.5) > 0
    bad = dict(config)
    bad["unsupported"] = True
    config_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown dispatcher"):
        ActiveInferenceFactory.create_from_yaml(config_path)


class TinyActiveModel(ActiveInferenceModel):
    def _load_config(self, config_path: str | Path | dict[str, object]) -> dict[str, object]:
        return {"prior_beliefs": [0.5, 0.5]}

    def _initialize_matrices(self) -> None:
        model = discrete_model()
        self.dispatcher = ActiveInferenceDispatcher(
            InferenceConfig("variational", "discrete", 1, 1.0, 1.0), model
        )

    def _initialize_state(self) -> ModelState:
        return ModelState(np.array([0.5, 0.5]), np.array([0.5, 0.5]), 1.0, 0.0, 0.0)

    def step(self, action: int | None = None) -> tuple[int, float]:
        return 0, self.calculate_free_energy()

    def visualize(self, plot_type: str, **kwargs: object) -> object:
        return {"plot_type": plot_type, "kwargs": kwargs}


def test_base_lifecycle_persistence_and_contracts(tmp_path: Path) -> None:
    model = TinyActiveModel({})
    model.update_beliefs(1)
    model.infer_policies()
    model.update_precision()
    assert np.isfinite(model.calculate_free_energy())
    saved = model.save_state(tmp_path / "base-state.yaml")
    model.load_state(saved)
    assert np.isclose(model.state.beliefs.sum(), 1.0)


def test_node_creation_relative_paths_and_network_aliases(tmp_path: Path) -> None:
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "agent_template.md").write_text(
        "# {{ agent_name }}\n", encoding="utf-8"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"templates": "templates", "knowledge_base": "knowledge"}}),
        encoding="utf-8",
    )
    output = NodeCreator(config_path).create_node("agent", "A safe node")
    assert output == tmp_path / "knowledge" / "agents" / "A safe node.md"
    with pytest.raises(ValueError):
        NodeCreator(config_path).create_node("agent", "../escape")
    (tmp_path / "knowledge" / "agents" / "linked.md").write_text(
        "---\ntype: agent\n---\n[[A safe node|display]]\n", encoding="utf-8"
    )
    graph_config = tmp_path / "graph.yaml"
    graph_config.write_text(
        yaml.safe_dump({"paths": {"knowledge_base": "knowledge"}, "visualization": {"seed": 0}}),
        encoding="utf-8",
    )
    graph = NetworkVisualizer(graph_config).build_network()
    assert len(graph) == 2
    assert graph.number_of_edges() == 1


def test_continuous_animation_has_multiple_frames(tmp_path: Path) -> None:
    agent = ContinuousActiveInference(seed=3)
    history = {"belief_means": [], "time": []}
    for _ in range(4):
        history["belief_means"].append(agent.state.belief_means.copy())
        history["time"].append(agent.state.time)
        agent.step(np.zeros(2))
    path = ContinuousVisualizer(tmp_path).save_animation(history, tmp_path / "beliefs.gif", fps=10)
    with Image.open(path) as image:
        assert image.n_frames == 4


def test_continuous_configuration_and_observation_validation() -> None:
    with pytest.raises(ValueError):
        ContinuousActiveInference(n_states=0)
    with pytest.raises(ValueError):
        ContinuousActiveInference(observation_matrix=np.eye(3))
    with pytest.raises(ValueError):
        ContinuousActiveInference(observation_precision=0.0)
    agent = ContinuousActiveInference(precision_learning_rate=0.5)
    with pytest.raises(ValueError):
        agent.step(np.zeros(3))
    agent.step(np.ones(2))


def test_path_like_links_are_explicit_and_benchmarks_are_json_serializable(tmp_path: Path) -> None:
    (tmp_path / "index.md").write_text(
        "[[tools/README]]\n[[concept_without_path]]\n", encoding="utf-8"
    )
    report = verify_link_report(tmp_path)
    assert len(report.broken_links) == 1
    assert report.broken_links[0]["target"] == "tools/README"
    benchmark_report = run_benchmarks(repetitions=1)
    json.dumps(benchmark_report)
