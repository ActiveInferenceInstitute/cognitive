"""Validated homeostatic control built on discrete Active Inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..matrices.matrix_ops import MatrixOps
from .base import ActiveInferenceModel, ModelState
from .dispatcher import ActiveInferenceDispatcher, InferenceConfig, InferenceMethod, PolicyType
from .generative_model import DiscreteGenerativeModel


@dataclass
class StateSpace:
    """One-dimensional labelled state space used by the homeostatic model."""

    dimensions: list[int]
    labels: dict[str, list[str]]
    mappings: dict[str, np.ndarray]
    hierarchical_levels: int | None = 1

    def validate(self) -> bool:
        if len(self.dimensions) != 1 or self.dimensions[0] < 1:
            return False
        if not self.labels:
            return False
        if any(len(values) != self.dimensions[0] for values in self.labels.values()):
            return False
        if not self.mappings:
            return False
        return all(
            isinstance(mapping, np.ndarray)
            and mapping.ndim == 2
            and mapping.shape == (self.dimensions[0], self.dimensions[0])
            and np.all(np.isfinite(mapping))
            for mapping in self.mappings.values()
        )


@dataclass
class ObservationModel:
    """Observation likelihood model for a discrete state space."""

    state_space: StateSpace
    observation_space: StateSpace
    likelihood_matrix: np.ndarray
    noise_model: str = "categorical"
    precision: float = 1.0

    def __post_init__(self) -> None:
        self.likelihood_matrix = np.asarray(self.likelihood_matrix, dtype=float)
        expected = (self.observation_space.dimensions[0], self.state_space.dimensions[0])
        if self.likelihood_matrix.shape != expected:
            raise ValueError(f"likelihood_matrix must have shape {expected}")
        if not np.all(np.isfinite(self.likelihood_matrix)) or np.any(self.likelihood_matrix < 0):
            raise ValueError("likelihood_matrix must be finite and non-negative")
        if not np.allclose(self.likelihood_matrix.sum(axis=0), 1.0):
            raise ValueError("likelihood_matrix must be column stochastic")
        if self.noise_model not in {"categorical", "gaussian"}:
            raise ValueError(f"Unsupported noise model: {self.noise_model}")
        if self.precision <= 0 or not np.isfinite(self.precision):
            raise ValueError("precision must be finite and positive")

    def compute_likelihood(self, observation: np.ndarray, state: np.ndarray) -> float:
        state_values = np.asarray(state, dtype=float).reshape(-1)
        values = np.asarray(observation, dtype=float).reshape(-1)
        if state_values.shape != (self.state_space.dimensions[0],):
            raise ValueError("state must match the state-space dimension")
        if values.shape != (self.observation_space.dimensions[0],):
            raise ValueError("observation must match the observation-space dimension")
        if np.any(state_values < 0) or not np.isclose(state_values.sum(), 1.0):
            raise ValueError("state must be a normalized non-negative distribution")
        prediction = self.likelihood_matrix @ state_values
        if self.noise_model == "categorical":
            return float(np.dot(values, prediction))
        return float(np.exp(-0.5 * self.precision * np.sum(np.square(values - prediction))))


@dataclass
class TransitionModel:
    """Column-stochastic state transitions indexed by action label."""

    state_space: StateSpace
    action_space: StateSpace
    transition_matrices: dict[str, np.ndarray]
    temporal_horizon: int
    control_modes: list[str] = field(default_factory=lambda: ["homeostatic"])

    def __post_init__(self) -> None:
        state_count = self.state_space.dimensions[0]
        expected_actions = self.action_space.labels.get("actions", [])
        if self.temporal_horizon < 1:
            raise ValueError("temporal_horizon must be at least one")
        if set(self.transition_matrices) != set(expected_actions):
            raise ValueError("A transition matrix is required for every action label")
        for label, matrix in self.transition_matrices.items():
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != (state_count, state_count):
                raise ValueError(f"Transition {label} must have shape {(state_count, state_count)}")
            if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
                raise ValueError(f"Transition {label} must be finite and non-negative")
            if not np.allclose(matrix.sum(axis=0), 1.0):
                raise ValueError(f"Transition {label} must be column stochastic")
            self.transition_matrices[label] = matrix

    def get_transition_matrix(self, action: int | str) -> np.ndarray:
        labels = self.action_space.labels["actions"]
        if isinstance(action, (int, np.integer)):
            index = int(action)
            if not 0 <= index < len(labels):
                raise ValueError(f"Action out of range: {action}")
            action = labels[index]
        if action not in self.transition_matrices:
            raise ValueError(f"Unknown action: {action}")
        return self.transition_matrices[action].copy()


class ControlMode(ABC):
    """Policy-prior interface for homeostatic controllers."""

    @abstractmethod
    def compute_policy_prior(self, state: ModelState, goal: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HomeostaticControl(ControlMode):
    """Prefer beliefs close to a configured target."""

    def __init__(self, bounds: tuple[float, float], target_state: str | int, weight: float = 1.0):
        if bounds[0] >= bounds[1]:
            raise ValueError("bounds must be increasing")
        if weight < 0 or not np.isfinite(weight):
            raise ValueError("weight must be finite and non-negative")
        self.bounds = bounds
        self.target_state = target_state
        self.weight = weight

    def compute_policy_prior(self, state: ModelState, goal: np.ndarray) -> np.ndarray:
        deviation = np.abs(np.asarray(state.beliefs) - np.asarray(goal))
        return np.exp(-self.weight * deviation)


class AdaptiveControl(ControlMode):
    """Balance target tracking with uncertainty-driven exploration."""

    def __init__(self, learning_rate: float = 0.1, exploration_weight: float = 0.3):
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 <= exploration_weight <= 1.0:
            raise ValueError("exploration_weight must be in [0, 1]")
        self.learning_rate = learning_rate
        self.exploration_weight = exploration_weight

    def compute_policy_prior(self, state: ModelState, goal: np.ndarray) -> np.ndarray:
        exploitation = -np.abs(state.beliefs - goal)
        exploration = -float(state.prediction_error) * np.ones_like(state.beliefs)
        logits = (
            1.0 - self.exploration_weight
        ) * exploitation + self.exploration_weight * exploration
        shifted = logits - np.max(logits)
        return np.exp(shifted)


class HomeostaticInference(ActiveInferenceModel):
    """Discrete homeostatic inference with validated configuration."""

    def __init__(self, config_path: str | Path | dict[str, Any], control_mode: ControlMode):
        self.control_mode = control_mode
        super().__init__(config_path)

    def _load_config(self, config_path: str | Path | dict[str, Any]) -> dict[str, Any]:
        if isinstance(config_path, dict):
            config = dict(config_path)
        else:
            with Path(config_path).open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file) or {}
        if not isinstance(config, dict):
            raise ValueError("Homeostatic configuration must be a mapping")
        required = {"state_spaces", "observation_model", "transition_model", "inference"}
        missing = required.difference(config)
        if missing:
            raise ValueError(f"Missing homeostatic configuration sections: {sorted(missing)}")
        allowed_root = required | {"target_state", "initial_beliefs", "action_prior", "seed"}
        unknown_root = set(config).difference(allowed_root)
        if unknown_root:
            raise ValueError(f"Unknown homeostatic configuration sections: {sorted(unknown_root)}")
        spaces = config["state_spaces"]
        if not isinstance(spaces, dict) or set(spaces) != {"environment", "observation", "action"}:
            raise ValueError("state_spaces must define environment, observation, and action")
        for name, raw_space in spaces.items():
            if not isinstance(raw_space, dict):
                raise ValueError(f"state_spaces.{name} must be a mapping")
            unknown = set(raw_space).difference(
                {"dimensions", "labels", "mappings", "hierarchical_levels"}
            )
            if unknown:
                raise ValueError(f"Unknown state_spaces.{name} fields: {sorted(unknown)}")
        observation_model = config["observation_model"]
        if not isinstance(observation_model, dict):
            raise ValueError("observation_model must be a mapping")
        unknown = set(observation_model).difference(
            {"likelihood_matrix", "noise_model", "precision"}
        )
        if unknown:
            raise ValueError(f"Unknown observation_model fields: {sorted(unknown)}")
        transition_model = config["transition_model"]
        if not isinstance(transition_model, dict):
            raise ValueError("transition_model must be a mapping")
        unknown = set(transition_model).difference(
            {"transition_matrices", "temporal_horizon", "control_modes"}
        )
        if unknown:
            raise ValueError(f"Unknown transition_model fields: {sorted(unknown)}")
        inference = config["inference"]
        if not isinstance(inference, dict):
            raise ValueError("inference must be a mapping")
        unknown = set(inference).difference(
            {
                "method",
                "policy_type",
                "temporal_horizon",
                "learning_rate",
                "precision_init",
                "num_samples",
                "temperature",
                "discount_factor",
                "exploration_weight",
                "policy_limit",
                "seed",
            }
        )
        if unknown:
            raise ValueError(f"Unknown inference fields: {sorted(unknown)}")
        return self._coerce_config(config)

    def _initialize_matrices(self) -> None:
        self._setup_spaces()
        inference = self.config["inference"]
        model_config = {
            "A": self.config["observation_model"]["likelihood_matrix"],
            "B": self._transition_tensor(),
            "C": self._preference_vector(),
            "D": self.config.get("initial_beliefs", np.ones(self.state_space.dimensions[0])),
            "E": self.config.get("action_prior", np.ones(self.action_space.dimensions[0])),
        }
        self.generative_model = DiscreteGenerativeModel.from_config(model_config)
        self.dispatcher = ActiveInferenceDispatcher(
            InferenceConfig(
                method=inference.get("method", InferenceMethod.VARIATIONAL.value),
                policy_type=inference.get("policy_type", PolicyType.DISCRETE.value),
                temporal_horizon=int(inference.get("temporal_horizon", 1)),
                learning_rate=float(inference.get("learning_rate", 1.0)),
                precision_init=float(inference.get("precision_init", 1.0)),
                num_samples=int(inference.get("num_samples", 256)),
                temperature=float(inference.get("temperature", 1.0)),
                discount_factor=float(inference.get("discount_factor", 0.95)),
                exploration_weight=float(inference.get("exploration_weight", 0.5)),
                policy_limit=int(inference.get("policy_limit", 4096)),
                seed=inference.get("seed", self.config.get("seed", 0)),
            ),
            self.generative_model,
        )
        self.matrix_ops = MatrixOps()
        observation_config = self.config["observation_model"]
        self.observation_model = ObservationModel(
            self.state_space,
            self.observation_space,
            np.asarray(observation_config["likelihood_matrix"], dtype=float),
            observation_config.get("noise_model", "categorical"),
            float(observation_config.get("precision", 1.0)),
        )
        self.transition_model = TransitionModel(
            self.state_space,
            self.action_space,
            {label: matrix for label, matrix in self._transition_by_label().items()},
            int(self.config["transition_model"].get("temporal_horizon", 1)),
            list(self.config["transition_model"].get("control_modes", ["homeostatic"])),
        )

    def _initialize_state(self) -> ModelState:
        state_count = self.state_space.dimensions[0]
        action_count = self.action_space.dimensions[0]
        beliefs = self._normalize(
            self.config.get("initial_beliefs", np.ones(state_count)), state_count
        )
        policies = self._normalize(
            self.config.get("action_prior", np.ones(action_count)), action_count
        )
        precision = float(self.config["inference"].get("precision_init", 1.0))
        return ModelState(beliefs, policies, precision, 0.0, 0.0)

    def _setup_spaces(self) -> None:
        spaces = self.config["state_spaces"]
        self.state_space = self._space_from_config(spaces, "environment")
        self.observation_space = self._space_from_config(spaces, "observation")
        self.action_space = self._space_from_config(spaces, "action")

    @staticmethod
    def _space_from_config(spaces: dict[str, Any], name: str) -> StateSpace:
        if name not in spaces:
            raise ValueError(f"Missing state space: {name}")
        raw = spaces[name]
        space = StateSpace(
            dimensions=[int(value) for value in raw["dimensions"]],
            labels={key: list(values) for key, values in raw["labels"].items()},
            mappings={
                key: np.asarray(value, dtype=float) for key, value in raw["mappings"].items()
            },
        )
        if not space.validate():
            raise ValueError(f"Invalid state space: {name}")
        return space

    def _transition_by_label(self) -> dict[str, np.ndarray]:
        labels = self.action_space.labels["actions"]
        configured = self.config["transition_model"].get("transition_matrices", {})
        state_count = self.state_space.dimensions[0]
        return {
            label: np.asarray(configured.get(label, np.eye(state_count)), dtype=float)
            for label in labels
        }

    def _transition_tensor(self) -> np.ndarray:
        matrices = self._transition_by_label()
        return np.stack([matrices[label] for label in self.action_space.labels["actions"]], axis=2)

    def _preference_vector(self) -> np.ndarray:
        target = self._target_vector()
        observation_count = self.observation_space.dimensions[0]
        if target.size == observation_count:
            return np.log(np.maximum(target, 1e-12))
        return np.zeros(observation_count)

    def update_beliefs(self, observation: int | np.ndarray) -> np.ndarray:
        prediction = self.generative_model.predict_observations(self.state.beliefs)
        posterior = self.dispatcher.dispatch_belief_update(observation, self.state)
        observed = self._observation_vector(observation, prediction.size)
        self.state.beliefs = posterior
        self.state.prediction_error = float(np.mean(np.square(observed - prediction)))
        self.state.free_energy = self._free_energy(self._target_vector())
        return posterior.copy()

    def step(self, action: int | None = None) -> tuple[int, float]:
        selected_action = self.select_action() if action is None else int(action)
        transition = self.transition_model.get_transition_matrix(selected_action)
        self.state.beliefs = self._normalize(
            transition @ self.state.beliefs, self.state_space.dimensions[0]
        )
        predicted_observation = self._normalize(
            self.observation_model.likelihood_matrix @ self.state.beliefs,
            self.observation_space.dimensions[0],
        )
        observation = int(np.argmax(predicted_observation))
        self.update_beliefs(observation)
        return observation, float(self.state.free_energy)

    def select_action(self) -> int:
        target = self._target_vector()
        scores = []
        for action_index in range(self.action_space.dimensions[0]):
            transition = self.transition_model.get_transition_matrix(action_index)
            predicted = self._normalize(transition @ self.state.beliefs, target.size)
            scores.append(float(np.linalg.norm(predicted - target, ord=1)))
        policy = self._normalize(
            self.matrix_ops.softmax(-np.asarray(scores) * self.state.precision)
        )
        control_prior = np.asarray(
            self.control_mode.compute_policy_prior(self.state, target), dtype=float
        )
        if control_prior.shape == policy.shape:
            policy = self._normalize(policy * self._normalize(control_prior))
        self.state.policies = policy
        return int(np.argmax(policy))

    def update_parameters(self, performance: dict[str, float]) -> None:
        if not isinstance(performance, dict):
            raise TypeError("performance must be a mapping")
        if isinstance(self.control_mode, AdaptiveControl):
            stability = float(performance.get("stability_index", 0.0))
            if not 0.0 <= stability <= 1.0:
                raise ValueError("stability_index must be in [0, 1]")
            self.control_mode.exploration_weight = float(np.clip(0.5 - 0.4 * stability, 0.1, 0.5))

    def reset(self) -> None:
        self.state = self._initialize_state()

    def visualize(self, plot_type: str, **kwargs: Any):
        if plot_type not in {"beliefs", "state"}:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        import matplotlib.pyplot as plt

        labels = next(iter(self.state_space.labels.values()))
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 4)))
        ax.bar(labels, self.state.beliefs)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Belief")
        ax.set_title("Homeostatic Belief State")
        fig.tight_layout()
        save_path = kwargs.get("save_path")
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
        return fig

    @staticmethod
    def _coerce_config(config: dict[str, Any]) -> dict[str, Any]:
        return config

    @staticmethod
    def _normalize(vector: list[float] | np.ndarray, size: int | None = None) -> np.ndarray:
        values = np.asarray(vector, dtype=float).reshape(-1)
        if size is not None and values.size != size:
            raise ValueError(f"Expected vector of size {size}, got {values.size}")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Probability vector must be finite and non-negative")
        total = float(values.sum())
        if total <= 1e-12:
            return np.full(size or values.size, 1.0 / (size or values.size))
        return values / total

    def _observation_vector(self, observation: int | np.ndarray, size: int) -> np.ndarray:
        if np.isscalar(observation):
            index = int(observation)
            if not 0 <= index < size:
                raise ValueError(f"Observation out of range: {index}")
            vector = np.zeros(size)
            vector[index] = 1.0
            return vector
        return self._normalize(observation, size)

    def _target_vector(self) -> np.ndarray:
        state_count = self.state_space.dimensions[0]
        target = self.config.get("target_state", getattr(self.control_mode, "target_state", None))
        if isinstance(target, (list, tuple, np.ndarray)):
            return self._normalize(target, state_count)
        if isinstance(target, (int, np.integer)):
            if not 0 <= int(target) < state_count:
                raise ValueError("target_state index is out of range")
            vector = np.zeros(state_count)
            vector[int(target)] = 1.0
            return vector
        if isinstance(target, str):
            labels = next(iter(self.state_space.labels.values()))
            if target not in labels:
                raise ValueError(f"Unknown target state: {target}")
            vector = np.zeros(state_count)
            vector[labels.index(target)] = 1.0
            return vector
        return np.full(state_count, 1.0 / state_count)

    def _free_energy(self, target: np.ndarray) -> float:
        complexity = MatrixOps.compute_kl_divergence(self.state.beliefs, target)
        accuracy = 0.5 * self.state.prediction_error
        return float(accuracy + complexity)


class HomeostaticFactory:
    """Factories for standard homeostatic control modes."""

    @staticmethod
    def create_basic(config_path: str | Path | dict[str, Any]) -> HomeostaticInference:
        target: str | int = 0
        if isinstance(config_path, dict):
            configured = config_path.get("target_state")
            if isinstance(configured, (str, int)):
                target = configured
        else:
            with Path(config_path).open(encoding="utf-8") as config_file:
                configured = yaml.safe_load(config_file) or {}
            if isinstance(configured, dict) and isinstance(
                configured.get("target_state"), (str, int)
            ):
                target = configured["target_state"]
        return HomeostaticInference(config_path, HomeostaticControl((-1.0, 1.0), target))

    @staticmethod
    def create_adaptive(config_path: str | Path | dict[str, Any]) -> HomeostaticInference:
        return HomeostaticInference(config_path, AdaptiveControl())
