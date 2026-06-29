"""
Homeostatic control implementation using Active Inference.
Provides abstractions and implementations for homeostatic control systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..matrices.matrix_ops import MatrixOps
from .base import ActiveInferenceModel, ModelState
from .dispatcher import (
    ActiveInferenceDispatcher,
    InferenceConfig,
    InferenceMethod,
    PolicyType,
)


@dataclass
class StateSpace:
    """Abstract representation of state spaces in active inference models."""
    dimensions: list[int]
    labels: dict[str, list[str]]
    mappings: dict[str, np.ndarray]
    hierarchical_levels: int | None = 1
    
    def validate(self) -> bool:
        """Validate state space configuration."""
        # Check dimensions match labels
        for dim, label_list in zip(self.dimensions, self.labels.values()):
            if len(label_list) != dim:
                return False
        return all(isinstance(mapping, np.ndarray) for mapping in self.mappings.values())

@dataclass
class ObservationModel:
    """Generalized observation model for active inference."""
    state_space: StateSpace
    observation_space: StateSpace
    likelihood_matrix: np.ndarray
    noise_model: str = "gaussian"
    precision: float = 1.0
    
    def compute_likelihood(self,
                         observation: np.ndarray,
                         state: np.ndarray) -> float:
        """Compute observation likelihood given state."""
        if self.noise_model == "gaussian":
            prediction = np.dot(self.likelihood_matrix, state)
            return np.exp(-0.5 * self.precision * np.sum(np.square(observation - prediction)))
        else:
            raise ValueError(f"Unsupported noise model: {self.noise_model}")

@dataclass
class TransitionModel:
    """Dynamic transition model for state evolution."""
    state_space: StateSpace
    action_space: StateSpace
    transition_matrices: dict[str, np.ndarray]
    temporal_horizon: int
    control_modes: list[str] = field(default_factory=lambda: ["homeostatic"])
    
    def get_transition_matrix(self, action: int | str) -> np.ndarray:
        """Get transition matrix for given action."""
        if isinstance(action, int):
            action = self.action_space.labels["actions"][action]
        return self.transition_matrices[action]

class ControlMode(ABC):
    """Abstract base class for control modes."""
    
    @abstractmethod
    def compute_policy_prior(self,
                           state: ModelState,
                           goal: np.ndarray) -> np.ndarray:
        """Compute policy prior based on control mode."""
        pass

class HomeostaticControl(ControlMode):
    """Homeostatic control mode implementation."""
    
    def __init__(self,
                 bounds: tuple[float, float],
                 target_state: str | int,
                 weight: float = 1.0):
        self.bounds = bounds
        self.target_state = target_state
        self.weight = weight
    
    def compute_policy_prior(self,
                           state: ModelState,
                           goal: np.ndarray) -> np.ndarray:
        """Compute homeostatic control policy prior."""
        deviation = np.abs(state.beliefs - goal)
        return np.exp(-self.weight * deviation)

HomestaticControl = HomeostaticControl

class AdaptiveControl(ControlMode):
    """Adaptive control mode implementation."""
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 exploration_weight: float = 0.3):
        self.learning_rate = learning_rate
        self.exploration_weight = exploration_weight
    
    def compute_policy_prior(self,
                           state: ModelState,
                           goal: np.ndarray) -> np.ndarray:
        """Compute adaptive control policy prior."""
        # Balance exploitation and exploration
        exploitation = -np.abs(state.beliefs - goal)
        exploration = -state.prediction_error * np.ones_like(state.beliefs)
        return np.exp(
            (1 - self.exploration_weight) * exploitation +
            self.exploration_weight * exploration
        )

class HomeostaticInference(ActiveInferenceModel):
    """Homeostatic control using Active Inference."""
    
    def __init__(self,
                 config_path: str | Path,
                 control_mode: ControlMode):
        self.config_path = config_path
        self.control_mode = control_mode
        super().__init__(config_path)

    def _load_config(self, config_path: str | Path) -> dict[str, Any]:
        if isinstance(config_path, dict):
            config = config_path
        else:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        return self._coerce_config(config)

    def _initialize_matrices(self):
        self.matrix_ops = MatrixOps()
        self._setup_spaces()
        inference_config = self.config.get("inference", {})
        self.dispatcher = ActiveInferenceDispatcher(
            InferenceConfig(
                method=InferenceMethod(
                    inference_config.get("method", InferenceMethod.VARIATIONAL.value)
                ),
                policy_type=PolicyType(
                    inference_config.get("policy_type", PolicyType.DISCRETE.value)
                ),
                temporal_horizon=int(inference_config.get("temporal_horizon", 1)),
                learning_rate=float(inference_config.get("learning_rate", 0.1)),
                precision_init=float(inference_config.get("precision_init", 1.0)),
                use_gpu=bool(inference_config.get("use_gpu", False)),
                num_samples=int(inference_config.get("num_samples", 1000)),
                temperature=float(inference_config.get("temperature", 1.0)),
                custom_params=inference_config.get("custom_params"),
            )
        )
        state_count = self.state_space.dimensions[0]
        observation_count = self.observation_space.dimensions[0]
        likelihood = np.asarray(
            self.config.get("observation_model", {}).get(
                "likelihood_matrix",
                np.eye(observation_count, state_count),
            ),
            dtype=float,
        )
        self.observation_model = ObservationModel(
            state_space=self.state_space,
            observation_space=self.observation_space,
            likelihood_matrix=likelihood,
            noise_model=self.config.get("observation_model", {}).get(
                "noise_model", "gaussian"
            ),
            precision=float(self.config.get("observation_model", {}).get("precision", 1.0)),
        )
        action_labels = self.action_space.labels.get(
            "actions",
            [str(index) for index in range(self.action_space.dimensions[0])],
        )
        configured_transitions = self.config.get("transition_model", {}).get(
            "transition_matrices", {}
        )
        transition_matrices = {
            label: np.asarray(configured_transitions.get(label, np.eye(state_count)), dtype=float)
            for label in action_labels
        }
        self.transition_model = TransitionModel(
            state_space=self.state_space,
            action_space=self.action_space,
            transition_matrices=transition_matrices,
            temporal_horizon=int(
                self.config.get("transition_model", {}).get("temporal_horizon", 1)
            ),
            control_modes=self.config.get("transition_model", {}).get(
                "control_modes", ["homeostatic"]
            ),
        )

    def _initialize_state(self) -> ModelState:
        state_count = self.state_space.dimensions[0]
        action_count = self.action_space.dimensions[0]
        beliefs = self._normalize(
            self.config.get("initial_beliefs", np.ones(state_count) / state_count),
            state_count,
        )
        policies = np.ones(action_count) / action_count
        precision = float(
            self.config.get("inference", {}).get(
                "precision_init",
                self.dispatcher.config.precision_init,
            )
        )
        return ModelState(
            beliefs=beliefs,
            policies=policies,
            precision=precision,
            free_energy=0.0,
            prediction_error=0.0,
        )
        
    def _setup_spaces(self):
        """Setup state spaces from configuration."""
        config = self.config
        self.state_space = StateSpace(
            dimensions=config["state_spaces"]["environment"]["dimensions"],
            labels=config["state_spaces"]["environment"]["labels"],
            mappings=config["state_spaces"]["environment"]["mappings"]
        )
        
        self.observation_space = StateSpace(
            dimensions=config["state_spaces"]["observation"]["dimensions"],
            labels=config["state_spaces"]["observation"]["labels"],
            mappings=config["state_spaces"]["observation"]["mappings"]
        )
        
        self.action_space = StateSpace(
            dimensions=config["state_spaces"]["action"]["dimensions"],
            labels=config["state_spaces"]["action"]["labels"],
            mappings=config["state_spaces"]["action"]["mappings"]
        )

    def update_beliefs(self, observation: int | np.ndarray) -> np.ndarray:
        """Update beliefs using active inference."""
        observation_vector = self._observation_vector(observation)
        state_likelihood = np.dot(
            self.observation_model.likelihood_matrix.T,
            observation_vector,
        )
        posterior = self._normalize(self.state.beliefs * state_likelihood)
        prediction = np.dot(self.observation_model.likelihood_matrix, self.state.beliefs)
        prediction_error = observation_vector - prediction
        self.state.beliefs = posterior
        self.state.prediction_error = float(np.mean(np.square(prediction_error)))
        self.state.free_energy = self._free_energy(self._target_vector())
        return posterior

    def step(self, action: int | None = None) -> tuple[int, float]:
        if action is None:
            action = self.select_action()
        transition = self.transition_model.get_transition_matrix(int(action))
        self.state.beliefs = self._normalize(
            np.dot(transition, self.state.beliefs),
            self.state_space.dimensions[0],
        )
        predicted_observation = self._normalize(
            np.dot(self.observation_model.likelihood_matrix, self.state.beliefs)
        )
        observation = int(np.argmax(predicted_observation))
        self.update_beliefs(observation)
        return observation, self.state.free_energy

    def visualize(self, plot_type: str, **kwargs):
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
            fig.savefig(save_path)
        return fig
    
    def select_action(self) -> int:
        """Select action using active inference."""
        target = self._target_vector()
        scores = []
        for action_index in range(self.action_space.dimensions[0]):
            transition = self.transition_model.get_transition_matrix(action_index)
            predicted_beliefs = self._normalize(
                np.dot(transition, self.state.beliefs),
                self.state_space.dimensions[0],
            )
            scores.append(np.linalg.norm(predicted_beliefs - target, ord=1))
        policy_prior = self.matrix_ops.softmax(-np.asarray(scores) * self.state.precision)
        control_prior = self.control_mode.compute_policy_prior(self.state, target)
        if control_prior.shape == policy_prior.shape:
            policy_prior = self._normalize(policy_prior * self._normalize(control_prior))
        self.state.policies = policy_prior
        return int(np.argmax(policy_prior))
    
    def update_parameters(self, performance: dict[str, float]):
        """Update model parameters based on performance."""
        if isinstance(self.control_mode, AdaptiveControl):
            # Update exploration weight based on performance
            stability = performance.get("stability_index", 0.0)
            self.control_mode.exploration_weight = np.clip(
                0.5 - 0.4 * stability,  # Reduce exploration as stability increases
                0.1,  # Minimum exploration
                0.5   # Maximum exploration
            )

    @staticmethod
    def _coerce_config(config: dict[str, Any]) -> dict[str, Any]:
        for space in config.get("state_spaces", {}).values():
            mappings = space.get("mappings", {})
            space["mappings"] = {
                key: np.asarray(value, dtype=float)
                for key, value in mappings.items()
            }
        return config

    @staticmethod
    def _normalize(vector: list[float] | np.ndarray, size: int | None = None) -> np.ndarray:
        values = np.asarray(vector, dtype=float).reshape(-1)
        if size is not None and values.size != size:
            raise ValueError(f"Expected vector of size {size}, got {values.size}")
        values = np.maximum(values, 0)
        total = np.sum(values)
        if total <= 1e-12:
            fallback_size = size or values.size
            return np.ones(fallback_size) / fallback_size
        return values / total

    def _observation_vector(self, observation: int | np.ndarray) -> np.ndarray:
        observation_count = self.observation_space.dimensions[0]
        if np.isscalar(observation):
            vector = np.zeros(observation_count)
            vector[int(observation)] = 1.0
            return vector
        return self._normalize(observation, observation_count)

    def _target_vector(self) -> np.ndarray:
        state_count = self.state_space.dimensions[0]
        target = self.config.get("target_state", getattr(self.control_mode, "target_state", None))
        if isinstance(target, (list, tuple, np.ndarray)):
            return self._normalize(target, state_count)
        if isinstance(target, (int, np.integer)):
            vector = np.zeros(state_count)
            vector[int(target)] = 1.0
            return vector
        if isinstance(target, str):
            for labels in self.state_space.labels.values():
                if target in labels:
                    vector = np.zeros(state_count)
                    vector[labels.index(target)] = 1.0
                    return vector
        return np.ones(state_count) / state_count

    def _free_energy(self, target: np.ndarray) -> float:
        complexity = MatrixOps.compute_kl_divergence(self.state.beliefs, target)
        accuracy = 0.5 * self.state.prediction_error
        return float(accuracy + complexity)

class HomeostaticFactory:
    """Factory for creating homeostatic control instances."""
    
    @staticmethod
    def create_basic(config_path: str | Path) -> HomeostaticInference:
        """Create basic homeostatic control instance."""
        control_mode = HomeostaticControl(
            bounds=(-1.0, 1.0),
            target_state="MEDIUM",
            weight=1.0
        )
        return HomeostaticInference(config_path, control_mode)
    
    @staticmethod
    def create_adaptive(config_path: str | Path) -> HomeostaticInference:
        """Create adaptive homeostatic control instance."""
        control_mode = AdaptiveControl(
            learning_rate=0.1,
            exploration_weight=0.3
        )
        return HomeostaticInference(config_path, control_mode)
