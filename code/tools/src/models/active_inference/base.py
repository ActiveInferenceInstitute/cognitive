"""Base contracts and state handling for Active Inference models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

EPS = 1e-12


@dataclass
class ModelState:
    """Serializable state shared by discrete Active Inference models."""

    beliefs: np.ndarray
    policies: np.ndarray
    precision: float
    free_energy: float
    prediction_error: float

    def copy(self) -> ModelState:
        return ModelState(
            beliefs=self.beliefs.copy(),
            policies=self.policies.copy(),
            precision=float(self.precision),
            free_energy=float(self.free_energy),
            prediction_error=float(self.prediction_error),
        )


class ActiveInferenceModel(ABC):
    """Abstract model lifecycle with validated state and persistence helpers."""

    STATE_SCHEMA_VERSION = 1

    def __init__(self, config_path: str | Path | dict[str, Any]):
        self.config = self._load_config(config_path)
        if not isinstance(self.config, dict):
            raise ValueError("Model configuration must be a mapping")
        self._initialize_matrices()
        self.state = self._initialize_state()
        self._validate_state(self.state)

    @abstractmethod
    def _load_config(self, config_path: str | Path | dict[str, Any]) -> dict[str, Any]:
        """Load and validate a concrete model configuration."""
        raise NotImplementedError

    @abstractmethod
    def _initialize_matrices(self) -> None:
        """Initialize the concrete model's generative model."""
        raise NotImplementedError

    @abstractmethod
    def _initialize_state(self) -> ModelState:
        """Initialize the concrete model's state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int | None = None) -> tuple[int, float]:
        """Advance the model and return an observation and free energy."""
        raise NotImplementedError

    @abstractmethod
    def visualize(self, plot_type: str, **kwargs: Any) -> Any:
        """Render a concrete model visualization."""
        raise NotImplementedError

    def update_beliefs(self, observation: int | np.ndarray) -> np.ndarray:
        """Update beliefs through the configured dispatcher.

        Concrete models with specialized observation semantics may override this
        method. Generic subclasses must expose an ``ActiveInferenceDispatcher``.
        """
        dispatcher = getattr(self, "dispatcher", None)
        if dispatcher is None:
            raise RuntimeError("The model must provide a dispatcher or override update_beliefs")
        beliefs = dispatcher.dispatch_belief_update(observation, self.state)
        prediction = self._generate_prediction()
        observation_vector = self._observation_vector(observation, prediction.size)
        prediction_error = observation_vector - prediction
        self.state.beliefs = beliefs
        self.state.prediction_error = float(np.mean(np.square(prediction_error)))
        self.calculate_free_energy()
        return beliefs.copy()

    def _generate_prediction(self) -> np.ndarray:
        """Generate an observation prediction from the configured likelihood."""
        model = getattr(self, "dispatcher", None)
        if model is not None and hasattr(model, "model"):
            return model.model.predict_observations(self.state.beliefs)
        matrix = np.asarray(self.config.get("generative_matrix"), dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != self.state.beliefs.size:
            raise RuntimeError("A dispatcher or a valid generative_matrix is required")
        prediction = self.state.beliefs @ matrix
        total = float(prediction.sum())
        return (
            prediction / total if total > EPS else np.full(prediction.shape, 1.0 / prediction.size)
        )

    def infer_policies(self) -> np.ndarray:
        """Infer and store the normalized first-action policy distribution."""
        dispatcher = getattr(self, "dispatcher", None)
        if dispatcher is None:
            raise RuntimeError("The model must provide a dispatcher to infer policies")
        self.state.policies = dispatcher.dispatch_policy_inference(self.state)
        return self.state.policies.copy()

    def _calculate_expected_free_energy(self) -> np.ndarray:
        """Return per-action expected free energy from the configured dispatcher."""
        dispatcher = getattr(self, "dispatcher", None)
        if dispatcher is None:
            raise RuntimeError(
                "The model must provide a dispatcher to calculate expected free energy"
            )
        return dispatcher._calculate_expected_free_energy(self.state)

    @staticmethod
    def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        if temperature <= 0 or not np.isfinite(temperature):
            raise ValueError("temperature must be finite and positive")
        logits = np.asarray(values, dtype=float)
        shifted = logits - np.max(logits)
        exponentials = np.exp(np.clip(shifted / temperature, -745.0, 709.0))
        return exponentials / exponentials.sum()

    def update_precision(self, beta: float = 0.9) -> float:
        """Update precision from the current prediction error."""
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if self.state.prediction_error < 0 or not np.isfinite(self.state.prediction_error):
            raise ValueError("prediction_error must be finite and non-negative")
        target = 1.0 / (self.state.prediction_error + 1e-8)
        self.state.precision = float(beta * self.state.precision + (1.0 - beta) * target)
        return self.state.precision

    def calculate_free_energy(self) -> float:
        """Calculate accuracy plus belief complexity relative to the model prior."""
        prior = np.asarray(self.config.get("prior_beliefs", self.state.beliefs.size), dtype=float)
        if prior.ndim == 0:
            prior = np.full(self.state.beliefs.shape, 1.0 / self.state.beliefs.size)
        if prior.shape != self.state.beliefs.shape:
            raise ValueError("prior_beliefs must match beliefs")
        prior = np.maximum(prior, 0.0)
        prior /= max(float(prior.sum()), EPS)
        beliefs = np.maximum(self.state.beliefs, 0.0)
        beliefs /= max(float(beliefs.sum()), EPS)
        complexity = float(np.sum(beliefs * np.log((beliefs + EPS) / (prior + EPS))))
        accuracy = 0.5 * float(self.state.prediction_error)
        self.state.free_energy = accuracy + complexity
        return self.state.free_energy

    def get_state(self) -> ModelState:
        return self.state.copy()

    def save_state(self, path: str | Path) -> Path:
        """Atomically save a versioned, human-readable state document."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "beliefs": self.state.beliefs.tolist(),
            "policies": self.state.policies.tolist(),
            "precision": float(self.state.precision),
            "free_energy": float(self.state.free_energy),
            "prediction_error": float(self.state.prediction_error),
        }
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("w", encoding="utf-8") as state_file:
            yaml.safe_dump(payload, state_file, sort_keys=True)
        temporary.replace(destination)
        return destination

    def load_state(self, path: str | Path) -> None:
        """Load and validate a previously saved state."""
        with Path(path).open(encoding="utf-8") as state_file:
            payload = yaml.safe_load(state_file)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != self.STATE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported or malformed model state")
        required = {"beliefs", "policies", "precision", "free_energy", "prediction_error"}
        if required.difference(payload):
            raise ValueError("Model state is missing required fields")
        candidate = ModelState(
            beliefs=np.asarray(payload["beliefs"], dtype=float),
            policies=np.asarray(payload["policies"], dtype=float),
            precision=float(payload["precision"]),
            free_energy=float(payload["free_energy"]),
            prediction_error=float(payload["prediction_error"]),
        )
        self._validate_state(candidate)
        self.state = candidate

    def _validate_state(self, state: ModelState) -> None:
        for name, values in (("beliefs", state.beliefs), ("policies", state.policies)):
            if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite, non-empty vector")
            if np.any(values < 0) or not np.isclose(values.sum(), 1.0):
                raise ValueError(f"{name} must be a normalized probability vector")
        for name, value in (
            ("precision", state.precision),
            ("free_energy", state.free_energy),
            ("prediction_error", state.prediction_error),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if state.precision <= 0 or state.prediction_error < 0:
            raise ValueError("precision must be positive and prediction_error non-negative")

    @staticmethod
    def _observation_vector(observation: int | np.ndarray, size: int) -> np.ndarray:
        if np.isscalar(observation):
            index = int(observation)
            if not 0 <= index < size:
                raise ValueError(f"Observation out of range: {index}")
            vector = np.zeros(size, dtype=float)
            vector[index] = 1.0
            return vector
        values = np.asarray(observation, dtype=float).reshape(-1)
        if values.shape != (size,) or np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"observation must be a finite, non-negative vector with shape {(size,)}"
            )
        total = float(values.sum())
        return values / total if total > EPS else np.full(size, 1.0 / size)
