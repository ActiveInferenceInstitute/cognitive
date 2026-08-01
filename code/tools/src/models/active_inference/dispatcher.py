"""Validated inference dispatch for discrete Active Inference models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .base import ModelState
from .generative_model import DiscreteGenerativeModel


class InferenceMethod(str, Enum):
    """Implemented belief and policy inference methods."""

    VARIATIONAL = "variational"
    SAMPLING = "sampling"
    MEAN_FIELD = "mean_field"


class PolicyType(str, Enum):
    """Policy representation supported by the discrete dispatcher."""

    DISCRETE = "discrete"


@dataclass
class InferenceConfig:
    """Validated configuration for inference dispatch."""

    method: InferenceMethod
    policy_type: PolicyType
    temporal_horizon: int
    learning_rate: float
    precision_init: float
    num_samples: int = 256
    temperature: float = 1.0
    discount_factor: float = 0.95
    exploration_weight: float = 0.5
    # If policy_limit truncates enumeration below one policy per first action,
    # dispatch_policy_inference raises instead of silently zeroing that action.
    policy_limit: int = 4096
    seed: int | None = 0
    custom_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.method = InferenceMethod(self.method)
        self.policy_type = PolicyType(self.policy_type)
        if self.policy_type is not PolicyType.DISCRETE:
            raise ValueError("Only discrete policies are implemented by this dispatcher")
        if self.temporal_horizon < 1:
            raise ValueError("temporal_horizon must be at least one")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.precision_init <= 0 or not np.isfinite(self.precision_init):
            raise ValueError("precision_init must be finite and positive")
        if self.num_samples < 2:
            raise ValueError("num_samples must be at least two")
        if self.temperature <= 0 or not np.isfinite(self.temperature):
            raise ValueError("temperature must be finite and positive")
        if not 0.0 < self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be in (0, 1]")
        if not 0.0 <= self.exploration_weight <= 1.0:
            raise ValueError("exploration_weight must be in [0, 1]")
        if self.policy_limit < 1:
            raise ValueError("policy_limit must be positive")
        self.custom_params = dict(self.custom_params)


class ActiveInferenceDispatcher:
    """Dispatch validated inference methods over one discrete generative model."""

    def __init__(
        self,
        config: InferenceConfig,
        model: DiscreteGenerativeModel,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self._rng = rng or np.random.default_rng(config.seed)

    def dispatch_belief_update(
        self,
        observation: int | np.ndarray,
        current_state: ModelState,
    ) -> np.ndarray:
        """Return a finite, normalized posterior without mutating ``current_state``."""
        update_fn = {
            InferenceMethod.VARIATIONAL: self._variational_belief_update,
            InferenceMethod.SAMPLING: self._sampling_belief_update,
            InferenceMethod.MEAN_FIELD: self._mean_field_belief_update,
        }[self.config.method]
        beliefs = update_fn(observation, current_state)
        return self._normalize(beliefs, self.model.num_states)

    def dispatch_policy_inference(
        self,
        state: ModelState,
        goal_prior: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the inferred distribution over the first action of each policy."""
        update_fn = {
            InferenceMethod.VARIATIONAL: self._variational_policy_inference,
            InferenceMethod.SAMPLING: self._sampling_policy_inference,
            InferenceMethod.MEAN_FIELD: self._mean_field_policy_inference,
        }[self.config.method]
        policies = update_fn(state, goal_prior)
        return self._normalize(policies, self.model.num_actions)

    def _variational_belief_update(
        self, observation: int | np.ndarray, state: ModelState
    ) -> np.ndarray:
        posterior = self.model.posterior(observation, state.beliefs)
        return self._blend(state.beliefs, posterior)

    def _mean_field_belief_update(
        self, observation: int | np.ndarray, state: ModelState
    ) -> np.ndarray:
        posterior = self.model.posterior(observation, state.beliefs)
        log_q = np.log(np.maximum(self._normalize(state.beliefs, self.model.num_states), 1e-12))
        log_posterior = np.log(np.maximum(posterior, 1e-12))
        logits = log_q + self.config.learning_rate * state.precision * (log_posterior - log_q)
        return self._softmax(logits)

    def _sampling_belief_update(
        self, observation: int | np.ndarray, state: ModelState
    ) -> np.ndarray:
        posterior = self.model.posterior(observation, state.beliefs)
        concentration = max(float(self.config.num_samples), 2.0)
        particles = self._rng.dirichlet(posterior * concentration, size=self.config.num_samples)
        return self._normalize(np.mean(particles, axis=0), self.model.num_states)

    def _variational_policy_inference(
        self, state: ModelState, goal_prior: np.ndarray | None
    ) -> np.ndarray:
        energies = self.calculate_expected_free_energy(state, goal_prior)
        return self._softmax(-energies / self.config.temperature)

    def _mean_field_policy_inference(
        self, state: ModelState, goal_prior: np.ndarray | None
    ) -> np.ndarray:
        # Compute the expected free energy without baking the action prior into
        # it, then apply the prior exactly once through the mean-field logits.
        # Passing goal_prior into calculate_expected_free_energy as well would
        # double-count it (the prior would enter the softmax twice).
        energies = self.calculate_expected_free_energy(state)
        prior = self._action_prior(goal_prior)
        logits = np.log(prior + 1e-12) - energies / self.config.temperature
        return self._softmax(logits)

    def _sampling_policy_inference(
        self, state: ModelState, goal_prior: np.ndarray | None
    ) -> np.ndarray:
        energies = self.calculate_expected_free_energy(state, goal_prior)
        target = self._softmax(-energies / self.config.temperature)
        samples = self._rng.choice(
            self.model.num_actions,
            size=self.config.num_samples,
            p=target,
        )
        counts = np.bincount(samples, minlength=self.model.num_actions).astype(float)
        return self._normalize(counts + 1e-6, self.model.num_actions)

    def calculate_expected_free_energy(
        self, state: ModelState, goal_prior: np.ndarray | None = None
    ) -> np.ndarray:
        """Return per-action expected free energy for the given beliefs.

        This public method lets model classes (such as
        ``ActiveInferenceModel``) compute expected free energy without reaching
        into private dispatcher internals.
        """
        policies = self.model.enumerate_policies(
            self.config.temporal_horizon,
            self.config.policy_limit,
        )
        values = np.zeros(len(policies), dtype=float)
        for index, policy in enumerate(policies):
            values[index] = self.model.evaluate_policy(
                state.beliefs,
                policy,
                self.config.discount_factor,
            )
        first_action_values = np.full(self.model.num_actions, np.inf, dtype=float)
        for action in range(self.model.num_actions):
            candidates = [
                value
                for policy, value in zip(policies, values, strict=False)
                if policy[0] == action
            ]
            if not candidates:
                raise ValueError(
                    f"policy_limit={self.config.policy_limit} is too low to enumerate a "
                    f"policy that starts with action {action}. Raise policy_limit or reduce "
                    "temporal_horizon; leaving this action implicit would silently give it "
                    "zero policy probability."
                )
            first_action_values[action] = min(candidates)
        if goal_prior is not None:
            prior = self._action_prior(goal_prior)
            first_action_values -= np.log(prior + 1e-12)
        return first_action_values

    def update_precision(self, prediction_error: float) -> float:
        """Update precision using a bounded reciprocal prediction error."""
        error = float(prediction_error)
        if error < 0 or not np.isfinite(error):
            raise ValueError("prediction_error must be finite and non-negative")
        target = 1.0 / (error + 1e-8)
        self.config.precision_init = float(
            np.clip(0.9 * self.config.precision_init + 0.1 * target, 0.1, 1e6)
        )
        return self.config.precision_init

    def _blend(self, prior: np.ndarray, posterior: np.ndarray) -> np.ndarray:
        prior_vector = self._normalize(prior, self.model.num_states)
        posterior_vector = self._normalize(posterior, self.model.num_states)
        return self._normalize(
            (1.0 - self.config.learning_rate) * prior_vector
            + self.config.learning_rate * posterior_vector,
            self.model.num_states,
        )

    def _action_prior(self, goal_prior: np.ndarray | None) -> np.ndarray:
        if goal_prior is None:
            return self.model.E.copy()
        candidate = np.asarray(goal_prior, dtype=float).reshape(-1)
        if candidate.shape != (self.model.num_actions,):
            raise ValueError(f"goal_prior must have shape {(self.model.num_actions,)}")
        return self._normalize(candidate, self.model.num_actions)

    @staticmethod
    def _normalize(values: np.ndarray, size: int) -> np.ndarray:
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size != size or not np.all(np.isfinite(array)):
            raise ValueError(f"Expected {size} finite values")
        array = np.maximum(array, 0.0)
        total = float(array.sum())
        if total <= 1e-12:
            return np.full(size, 1.0 / size)
        return array / total

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        logits = np.asarray(values, dtype=float)
        shifted = logits - np.max(logits)
        exp_values = np.exp(np.clip(shifted, -745.0, 709.0))
        return exp_values / exp_values.sum()


class ActiveInferenceFactory:
    """Factory for configured discrete Active Inference dispatchers."""

    @staticmethod
    def create(
        config: InferenceConfig, model: DiscreteGenerativeModel
    ) -> ActiveInferenceDispatcher:
        return ActiveInferenceDispatcher(config, model)

    @staticmethod
    def create_from_yaml(config_path: str | Path) -> ActiveInferenceDispatcher:
        path = Path(config_path)
        with path.open(encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file) or {}
        if not isinstance(raw, dict):
            raise ValueError("Dispatcher configuration must be a mapping")
        model_config = raw.get("generative_model", raw.get("model"))
        if not isinstance(model_config, dict):
            raise ValueError("Configuration must contain a generative_model mapping")
        model = DiscreteGenerativeModel.from_config(model_config)
        config_fields = {
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
            "custom_params",
        }
        config_values = {key: value for key, value in raw.items() if key in config_fields}
        missing = config_fields.difference(config_values).difference(
            {
                "num_samples",
                "temperature",
                "discount_factor",
                "exploration_weight",
                "policy_limit",
                "seed",
                "custom_params",
            }
        )
        if missing:
            raise ValueError(f"Missing dispatcher configuration fields: {sorted(missing)}")
        unknown = set(raw).difference(config_fields | {"generative_model", "model"})
        if unknown:
            raise ValueError(f"Unknown dispatcher configuration fields: {sorted(unknown)}")
        return ActiveInferenceFactory.create(InferenceConfig(**config_values), model)
