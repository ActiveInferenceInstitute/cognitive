"""Lifecycle, persistence, and configuration coverage for the model classes.

``test_truth_audit.py`` exercises the happy paths of the public API. This module
covers the validation branches those paths never reach -- the guards that decide
whether a malformed configuration or a corrupted state file is rejected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from cognitive.models.active_inference.base import ActiveInferenceModel, ModelState
from cognitive.models.active_inference.dispatcher import (
    ActiveInferenceDispatcher,
    InferenceConfig,
)
from cognitive.models.active_inference.generative_model import DiscreteGenerativeModel
from cognitive.models.active_inference.homeostatic import (
    AdaptiveControl,
    HomeostaticControl,
    HomeostaticFactory,
    HomeostaticInference,
    ObservationModel,
    StateSpace,
    TransitionModel,
)


def two_state_model() -> DiscreteGenerativeModel:
    return DiscreteGenerativeModel(
        A=np.array([[0.9, 0.1], [0.1, 0.9]]),
        B=np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2),
        C=np.array([0.0, 1.0]),
        D=np.array([0.5, 0.5]),
        E=np.array([0.5, 0.5]),
    )


class DispatcherModel(ActiveInferenceModel):
    """Minimal concrete model that routes everything through the dispatcher."""

    def __init__(self, config: dict[str, object]) -> None:
        super().__init__(config)

    def _load_config(self, config_path):
        return dict(config_path)

    def _initialize_matrices(self) -> None:
        self.generative_model = two_state_model()
        self.dispatcher = ActiveInferenceDispatcher(
            InferenceConfig(
                method="variational",
                policy_type="discrete",
                temporal_horizon=2,
                learning_rate=0.5,
                precision_init=1.0,
                seed=0,
            ),
            self.generative_model,
        )

    def _initialize_state(self) -> ModelState:
        return ModelState(
            self.generative_model.D.copy(), self.generative_model.E.copy(), 1.0, 0.0, 0.0
        )

    def step(self, action: int | None = None) -> tuple[int, float]:
        self.update_beliefs(0)
        return 0, float(self.state.free_energy)

    def visualize(self, plot_type: str, **kwargs):
        return plot_type


class MatrixOnlyModel(DispatcherModel):
    """Model without a dispatcher, predicting through ``generative_matrix``."""

    def _initialize_matrices(self) -> None:
        self.generative_model = two_state_model()

    def _generate_prediction(self):
        return ActiveInferenceModel._generate_prediction(self)


def test_dispatcher_model_updates_beliefs_and_energy() -> None:
    """A concrete model normalizes beliefs and reports a finite free energy."""
    model = DispatcherModel({"prior_beliefs": [0.5, 0.5]})
    beliefs = model.update_beliefs(1)
    assert beliefs == pytest.approx(model.state.beliefs)
    assert np.isclose(beliefs.sum(), 1.0)
    assert np.isfinite(model.state.free_energy)
    assert model.state.prediction_error >= 0.0
    policies = model.infer_policies()
    assert np.isclose(policies.sum(), 1.0)
    energies = model._calculate_expected_free_energy()
    assert energies.shape == (2,)
    assert np.all(np.isfinite(energies))


def test_free_energy_is_complexity_plus_inaccuracy() -> None:
    """The reported free energy matches its definition term by term."""
    model = DispatcherModel({"prior_beliefs": [0.5, 0.5]})
    model.state.beliefs = np.array([0.8, 0.2])
    model.state.prediction_error = 0.25
    prior = np.array([0.5, 0.5])
    complexity = float(np.sum(model.state.beliefs * np.log(model.state.beliefs / prior)))
    assert model.calculate_free_energy() == pytest.approx(complexity + 0.5 * 0.25, abs=1e-9)
    # A belief equal to the prior contributes no complexity.
    model.state.beliefs = prior.copy()
    model.state.prediction_error = 0.0
    assert model.calculate_free_energy() == pytest.approx(0.0, abs=1e-9)


def test_free_energy_rejects_a_mismatched_prior() -> None:
    model = DispatcherModel({"prior_beliefs": [0.2, 0.3, 0.5]})
    with pytest.raises(ValueError, match="prior_beliefs must match beliefs"):
        model.calculate_free_energy()


def test_precision_update_is_bounded_and_validated() -> None:
    """Precision tracks the reciprocal error and rejects impossible inputs."""
    model = DispatcherModel({})
    model.state.prediction_error = 1.0
    updated = model.update_precision(beta=0.5)
    assert updated == pytest.approx(0.5 * 1.0 + 0.5 * (1.0 / (1.0 + 1e-8)), rel=1e-6)
    with pytest.raises(ValueError, match="beta must be in"):
        model.update_precision(beta=1.5)
    model.state.prediction_error = -1.0
    with pytest.raises(ValueError, match="prediction_error"):
        model.update_precision()


def test_state_round_trips_through_disk(tmp_path: Path) -> None:
    """Saved state reloads identically and rejects a version it cannot read."""
    model = DispatcherModel({})
    model.update_beliefs(1)
    saved = model.save_state(tmp_path / "nested" / "state.yaml")
    assert saved.is_file()
    before = model.get_state()

    model.state.beliefs = np.array([0.99, 0.01])
    model.load_state(saved)
    assert model.state.beliefs == pytest.approx(before.beliefs)
    assert model.state.precision == pytest.approx(before.precision)

    payload = yaml.safe_load(saved.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    saved.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported or malformed model state"):
        model.load_state(saved)

    payload["schema_version"] = ActiveInferenceModel.STATE_SCHEMA_VERSION
    del payload["precision"]
    saved.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        model.load_state(saved)

    payload["precision"] = -1.0
    saved.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="precision must be positive"):
        model.load_state(saved)

    payload["precision"] = 1.0
    payload["beliefs"] = [0.5, 0.9]
    saved.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="normalized probability vector"):
        model.load_state(saved)


def test_model_without_a_dispatcher_requires_a_generative_matrix() -> None:
    """The fallback prediction path is validated rather than silently wrong."""
    model = MatrixOnlyModel({"generative_matrix": [[0.8, 0.2], [0.3, 0.7]]})
    prediction = model._generate_prediction()
    assert prediction == pytest.approx(np.array([0.5, 0.5]) @ np.array([[0.8, 0.2], [0.3, 0.7]]))
    with pytest.raises(RuntimeError, match="must provide a dispatcher"):
        model.update_beliefs(0)
    with pytest.raises(RuntimeError, match="dispatcher to infer policies"):
        model.infer_policies()
    with pytest.raises(RuntimeError, match="calculate expected free energy"):
        model._calculate_expected_free_energy()

    broken = MatrixOnlyModel({"generative_matrix": [[0.8, 0.2, 0.0]]})
    with pytest.raises(RuntimeError, match="valid generative_matrix"):
        broken._generate_prediction()


def test_observation_vector_accepts_indices_and_distributions() -> None:
    """Both observation encodings normalize, and invalid ones are rejected."""
    assert ActiveInferenceModel._observation_vector(1, 3) == pytest.approx([0.0, 1.0, 0.0])
    assert ActiveInferenceModel._observation_vector(np.array([2.0, 2.0]), 2) == pytest.approx(
        [0.5, 0.5]
    )
    assert ActiveInferenceModel._observation_vector(np.zeros(2), 2) == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError, match="Observation out of range"):
        ActiveInferenceModel._observation_vector(5, 3)
    with pytest.raises(ValueError, match="finite, non-negative vector"):
        ActiveInferenceModel._observation_vector(np.array([-1.0, 1.0]), 2)


def test_base_softmax_validates_temperature() -> None:
    assert ActiveInferenceModel._softmax(np.zeros(3)) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    with pytest.raises(ValueError, match="temperature"):
        ActiveInferenceModel._softmax(np.zeros(3), temperature=0.0)


def homeostatic_config(states: int = 3, observations: int = 3) -> dict[str, object]:
    """A valid homeostatic configuration with independently sized spaces."""
    likelihood = np.full((observations, states), 1.0 / observations)
    likelihood[0, 0] = likelihood[0, 0] + 0.1
    likelihood[:, 0] = likelihood[:, 0] / likelihood[:, 0].sum()
    return {
        "state_spaces": {
            "environment": {
                "dimensions": [states],
                "labels": {"states": [f"s{index}" for index in range(states)]},
                "mappings": {"identity": np.eye(states).tolist()},
            },
            "observation": {
                "dimensions": [observations],
                "labels": {"observations": [f"o{index}" for index in range(observations)]},
                "mappings": {"identity": np.eye(observations).tolist()},
            },
            "action": {
                "dimensions": [2],
                "labels": {"actions": ["hold", "move"]},
                "mappings": {"identity": np.eye(2).tolist()},
            },
        },
        "observation_model": {"likelihood_matrix": likelihood.tolist()},
        "transition_model": {
            "transition_matrices": {
                "hold": np.eye(states).tolist(),
                "move": np.roll(np.eye(states), 1, axis=0).tolist(),
            },
            "temporal_horizon": 1,
        },
        "inference": {
            "method": "variational",
            "policy_type": "discrete",
            "temporal_horizon": 1,
            "learning_rate": 0.5,
            "precision_init": 1.0,
        },
        "target_state": "s0",
    }


@pytest.mark.parametrize(
    ("states", "observations"),
    [(3, 3), (3, 2), (2, 4)],
)
def test_homeostatic_preferences_are_defined_for_unequal_spaces(
    states: int, observations: int
) -> None:
    """Observation preferences come from the target pushed through ``A``.

    Reading the state target directly as an observation preference is only
    defined when the two spaces have equal size; every other shape silently
    fell back to a flat preference before this was fixed.
    """
    model = HomeostaticFactory.create_basic(homeostatic_config(states, observations))
    preferences = model._preference_vector()
    assert preferences.shape == (observations,)
    likelihood = np.asarray(model.config["observation_model"]["likelihood_matrix"], dtype=float)
    expected = likelihood @ model._target_vector()
    assert np.exp(preferences) == pytest.approx(expected / expected.sum(), abs=1e-9)
    # A non-uniform target must produce non-uniform preferences.
    assert not np.allclose(preferences, preferences[0])


@pytest.mark.parametrize("factory", ["create_basic", "create_adaptive"])
def test_homeostatic_models_step_and_select_actions(factory: str) -> None:
    """Both control modes drive a full step for unequal state and action counts."""
    model = getattr(HomeostaticFactory, factory)(homeostatic_config(3, 2))
    observation, free_energy = model.step()
    assert 0 <= observation < 2
    assert np.isfinite(free_energy)
    assert np.isclose(model.state.policies.sum(), 1.0)
    assert model.state.policies.shape == (2,)
    action = model.select_action()
    assert 0 <= action < 2
    model.reset()
    assert np.isclose(model.state.beliefs.sum(), 1.0)


def test_homeostatic_control_prior_shapes_are_reduced_per_action() -> None:
    """A state-shaped control prior is scored per action rather than dropped.

    With three states and two actions the two vectors cannot be multiplied, and
    the earlier implementation skipped the control mode entirely without saying
    so. The prior must now influence the policy.
    """
    config = homeostatic_config(3, 3)
    weak = HomeostaticInference(config, HomeostaticControl((-1.0, 1.0), "s0", weight=0.0))
    strong = HomeostaticInference(config, HomeostaticControl((-1.0, 1.0), "s0", weight=50.0))
    weak.state.beliefs = np.array([0.1, 0.6, 0.3])
    strong.state.beliefs = np.array([0.1, 0.6, 0.3])
    weak.select_action()
    strong.select_action()
    assert not np.allclose(weak.state.policies, strong.state.policies)


def test_homeostatic_rejects_a_control_mode_returning_invalid_scores() -> None:
    """An invalid control prior raises instead of being silently ignored."""

    class NegativeControl(HomeostaticControl):
        def compute_policy_prior(self, state, goal):
            return np.array([-1.0, -1.0, -1.0])

    model = HomeostaticInference(homeostatic_config(3, 3), NegativeControl((-1.0, 1.0), "s0"))
    with pytest.raises(ValueError, match="finite, non-negative"):
        model.select_action()


def test_homeostatic_configuration_errors_are_specific() -> None:
    """Each configuration guard names the section it rejected."""
    with pytest.raises(ValueError, match="Missing homeostatic configuration sections"):
        HomeostaticFactory.create_basic({"inference": {}})

    config = homeostatic_config()
    config["unexpected"] = 1
    with pytest.raises(ValueError, match="Unknown homeostatic configuration sections"):
        HomeostaticFactory.create_basic(config)

    config = homeostatic_config()
    del config["transition_model"]["transition_matrices"]["move"]
    with pytest.raises(ValueError, match="transition matrix is required"):
        HomeostaticFactory.create_basic(config)

    config = homeostatic_config()
    config["inference"]["unknown_field"] = 1
    with pytest.raises(ValueError, match="Unknown inference fields"):
        HomeostaticFactory.create_basic(config)

    config = homeostatic_config()
    config["observation_model"]["unknown_field"] = 1
    with pytest.raises(ValueError, match="Unknown observation_model fields"):
        HomeostaticFactory.create_basic(config)

    config = homeostatic_config()
    config["state_spaces"]["action"]["labels"]["actions"] = []
    with pytest.raises(ValueError, match="non-empty list"):
        HomeostaticFactory.create_basic(config)


def test_homeostatic_target_accepts_index_vector_and_label() -> None:
    """Every documented target encoding resolves to the same distribution."""
    for target, expected in (
        ("s1", [0.0, 1.0, 0.0]),
        (1, [0.0, 1.0, 0.0]),
        ([0.0, 2.0, 0.0], [0.0, 1.0, 0.0]),
    ):
        config = homeostatic_config()
        config["target_state"] = target
        model = HomeostaticFactory.create_adaptive(config)
        assert model._target_vector() == pytest.approx(expected)

    config = homeostatic_config()
    config["target_state"] = "not-a-state"
    with pytest.raises(ValueError, match="Unknown target state"):
        HomeostaticFactory.create_adaptive(config)._target_vector()

    config = homeostatic_config()
    config["target_state"] = 99
    with pytest.raises(ValueError, match="out of range"):
        HomeostaticFactory.create_adaptive(config)._target_vector()


def test_adaptive_control_parameters_respond_to_reported_stability() -> None:
    """Reported stability moves exploration within its documented bounds."""
    model = HomeostaticFactory.create_adaptive(homeostatic_config())
    model.update_parameters({"stability_index": 1.0})
    assert model.control_mode.exploration_weight == pytest.approx(0.1)
    model.update_parameters({"stability_index": 0.0})
    assert model.control_mode.exploration_weight == pytest.approx(0.5)
    with pytest.raises(ValueError, match="stability_index"):
        model.update_parameters({"stability_index": 2.0})
    with pytest.raises(TypeError, match="performance must be a mapping"):
        model.update_parameters([])
    # A non-adaptive control mode is unaffected rather than mutated.
    basic = HomeostaticFactory.create_basic(homeostatic_config())
    basic.update_parameters({"stability_index": 1.0})


def test_homeostatic_visualization_writes_a_figure(tmp_path: Path) -> None:
    model = HomeostaticFactory.create_basic(homeostatic_config())
    destination = tmp_path / "plots" / "beliefs.png"
    figure = model.visualize("beliefs", save_path=destination)
    assert destination.is_file()
    assert figure is not None
    with pytest.raises(ValueError, match="Unsupported plot type"):
        model.visualize("nonexistent")


def test_state_space_and_component_validation() -> None:
    """The declarative model components reject malformed inputs."""
    valid = StateSpace([2], {"states": ["a", "b"]}, {"identity": np.eye(2)})
    assert valid.validate()
    assert not StateSpace([0], {"states": []}, {"identity": np.eye(2)}).validate()
    assert not StateSpace([2], {}, {"identity": np.eye(2)}).validate()
    assert not StateSpace([2], {"states": ["a"]}, {"identity": np.eye(2)}).validate()
    assert not StateSpace([2], {"states": ["a", "b"]}, {}).validate()
    assert not StateSpace([2], {"states": ["a", "b"]}, {"identity": np.eye(3)}).validate()

    observations = StateSpace([2], {"observations": ["x", "y"]}, {"identity": np.eye(2)})
    likelihood = np.array([[0.9, 0.2], [0.1, 0.8]])
    observation_model = ObservationModel(valid, observations, likelihood)
    assert observation_model.compute_likelihood(
        np.array([1.0, 0.0]), np.array([1.0, 0.0])
    ) == pytest.approx(0.9)
    gaussian = ObservationModel(valid, observations, likelihood, noise_model="gaussian")
    assert 0.0 < gaussian.compute_likelihood(np.array([1.0, 0.0]), np.array([1.0, 0.0])) <= 1.0
    with pytest.raises(ValueError, match="Unsupported noise model"):
        ObservationModel(valid, observations, likelihood, noise_model="poisson")
    with pytest.raises(ValueError, match="column stochastic"):
        ObservationModel(valid, observations, np.array([[0.9, 0.2], [0.5, 0.8]]))
    with pytest.raises(ValueError, match="normalized non-negative"):
        observation_model.compute_likelihood(np.array([1.0, 0.0]), np.array([0.4, 0.4]))

    actions = StateSpace([2], {"actions": ["hold", "move"]}, {"identity": np.eye(2)})
    transitions = TransitionModel(
        valid, actions, {"hold": np.eye(2), "move": np.array([[0.0, 1.0], [1.0, 0.0]])}, 1
    )
    assert np.array_equal(transitions.get_transition_matrix("hold"), np.eye(2))
    assert np.array_equal(
        transitions.get_transition_matrix(1), np.array([[0.0, 1.0], [1.0, 0.0]])
    )
    with pytest.raises(ValueError, match="Action out of range"):
        transitions.get_transition_matrix(9)
    with pytest.raises(ValueError, match="Unknown action"):
        transitions.get_transition_matrix("fly")
    with pytest.raises(ValueError, match="temporal_horizon"):
        TransitionModel(valid, actions, {"hold": np.eye(2), "move": np.eye(2)}, 0)
    with pytest.raises(ValueError, match="required for every action"):
        TransitionModel(valid, actions, {"hold": np.eye(2)}, 1)


def test_control_mode_constructors_validate_their_parameters() -> None:
    with pytest.raises(ValueError, match="bounds must be increasing"):
        HomeostaticControl((1.0, 0.0), 0)
    with pytest.raises(ValueError, match="weight must be finite"):
        HomeostaticControl((0.0, 1.0), 0, weight=-1.0)
    with pytest.raises(ValueError, match="learning_rate"):
        AdaptiveControl(learning_rate=0.0)
    with pytest.raises(ValueError, match="exploration_weight"):
        AdaptiveControl(exploration_weight=1.5)
