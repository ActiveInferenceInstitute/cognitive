"""Validated discrete generative models used by Active Inference inference."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

EPS = 1e-12


def _as_probability_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be a finite, non-negative, non-empty vector")
    total = float(array.sum())
    if total <= EPS:
        raise ValueError(f"{name} must contain positive mass")
    return array / total


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(array.sum())
    if total <= EPS:
        return np.full(array.shape, 1.0 / array.size, dtype=float)
    return array / total


def _entropy(distribution: np.ndarray) -> float:
    probabilities = _normalize(distribution)
    return float(-np.sum(probabilities * np.log(probabilities + EPS)))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_normalized = _normalize(p)
    q_normalized = _normalize(q)
    return float(np.sum(p_normalized * np.log((p_normalized + EPS) / (q_normalized + EPS))))


@dataclass(frozen=True)
class DiscreteGenerativeModel:
    """A validated discrete POMDP generative model.

    Matrix conventions are explicit and shared by all discrete inference methods:

    * ``A[o, s] = P(o | s)``;
    * ``B[s_next, s_prev, a] = P(s_next | s_prev, a)``;
    * ``C[o]`` contains log preferences over observations;
    * ``D[s]`` is the prior over states;
    * ``E[a]`` is the prior over actions.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    E: np.ndarray

    def __post_init__(self) -> None:
        matrices = {name: np.asarray(value, dtype=float) for name, value in self.__dict__.items()}
        object.__setattr__(self, "A", matrices["A"])
        object.__setattr__(self, "B", matrices["B"])
        object.__setattr__(self, "C", matrices["C"].reshape(-1))
        object.__setattr__(self, "D", matrices["D"].reshape(-1))
        object.__setattr__(self, "E", matrices["E"].reshape(-1))

        if self.A.ndim != 2:
            raise ValueError("A must have shape (observations, states)")
        if self.B.ndim != 3:
            raise ValueError("B must have shape (states, states, actions)")
        if self.B.shape[0] != self.B.shape[1]:
            raise ValueError("B must have equal current and next state dimensions")
        observations, states = self.A.shape
        if self.B.shape[0] != states:
            raise ValueError("A and B state dimensions must agree")
        if self.C.shape != (observations,):
            raise ValueError("C must contain one preference per observation")
        if self.D.shape != (states,):
            raise ValueError("D must contain one prior per state")
        if self.E.shape != (self.B.shape[2],):
            raise ValueError("E must contain one prior per action")
        for name, matrix in (("A", self.A), ("B", self.B)):
            if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.allclose(self.A.sum(axis=0), 1.0):
            raise ValueError("A must be column stochastic")
        if not np.allclose(self.B.sum(axis=0), 1.0):
            raise ValueError("Each B[:, :, action] column must sum to one")
        object.__setattr__(self, "D", _as_probability_vector(self.D, name="D"))
        object.__setattr__(self, "E", _as_probability_vector(self.E, name="E"))
        if not np.all(np.isfinite(self.C)):
            raise ValueError("C must contain finite log preferences")

    @classmethod
    def from_config(cls, config: dict[str, object]) -> DiscreteGenerativeModel:
        """Construct a model from a mapping containing A/B/C/D/E arrays."""
        required = {"A", "B", "C", "D", "E"}
        missing = required.difference(config)
        if missing:
            raise ValueError(f"Missing generative model fields: {sorted(missing)}")
        unknown = set(config).difference(required)
        if unknown:
            raise ValueError(f"Unknown generative model fields: {sorted(unknown)}")
        return cls(*(np.asarray(config[name], dtype=float) for name in ("A", "B", "C", "D", "E")))

    @property
    def num_observations(self) -> int:
        return self.A.shape[0]

    @property
    def num_states(self) -> int:
        return self.A.shape[1]

    @property
    def num_actions(self) -> int:
        return self.B.shape[2]

    @property
    def preferences(self) -> np.ndarray:
        shifted = self.C - np.max(self.C)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum()

    def predict_states(self, beliefs: np.ndarray, action: int) -> np.ndarray:
        action_index = self._validate_action(action)
        prior = _as_probability_vector(beliefs, name="beliefs")
        if prior.shape != self.D.shape:
            raise ValueError(f"beliefs must have shape {self.D.shape}")
        return _normalize(self.B[:, :, action_index] @ prior)

    def predict_observations(self, beliefs: np.ndarray) -> np.ndarray:
        prior = _as_probability_vector(beliefs, name="beliefs")
        if prior.shape != self.D.shape:
            raise ValueError(f"beliefs must have shape {self.D.shape}")
        return _normalize(self.A @ prior)

    def posterior(self, observation: int | np.ndarray, prior: np.ndarray) -> np.ndarray:
        prior_vector = _as_probability_vector(prior, name="prior")
        if prior_vector.shape != self.D.shape:
            raise ValueError(f"prior must have shape {self.D.shape}")
        if np.isscalar(observation):
            index = int(observation)
            if not 0 <= index < self.num_observations:
                raise ValueError(f"Observation out of range: {index}")
            likelihood = self.A[index]
        else:
            observation_vector = _as_probability_vector(observation, name="observation")
            if observation_vector.shape != (self.num_observations,):
                raise ValueError(f"observation must have shape {(self.num_observations,)}")
            likelihood = self.A.T @ observation_vector
        posterior = prior_vector * np.maximum(likelihood, 0.0)
        if float(posterior.sum()) <= EPS:
            return prior_vector
        return _normalize(posterior)

    def expected_free_energy(
        self, beliefs: np.ndarray, action: int
    ) -> tuple[float, float, float, float]:
        """Return total EFE, risk, ambiguity, and epistemic information gain."""
        predicted_states = self.predict_states(beliefs, action)
        predicted_observations = self.predict_observations(predicted_states)
        risk = _kl_divergence(predicted_observations, self.preferences)
        ambiguity = float(
            sum(
                predicted_states[state] * _entropy(self.A[:, state])
                for state in range(self.num_states)
            )
        )
        posterior_entropies = []
        for observation in range(self.num_observations):
            posterior = self.posterior(observation, predicted_states)
            posterior_entropies.append(predicted_observations[observation] * _entropy(posterior))
        epistemic = max(_entropy(predicted_states) - float(sum(posterior_entropies)), 0.0)
        total = risk + ambiguity - epistemic
        return float(total), float(risk), float(ambiguity), float(epistemic)

    def enumerate_policies(self, horizon: int, limit: int | None = None) -> list[tuple[int, ...]]:
        if horizon < 1:
            raise ValueError("horizon must be at least one")
        policies = product(range(self.num_actions), repeat=horizon)
        if limit is None:
            return list(policies)
        if limit < 1:
            raise ValueError("policy limit must be positive")
        return [policy for index, policy in enumerate(policies) if index < limit]

    def evaluate_policy(
        self, beliefs: np.ndarray, policy: tuple[int, ...], discount_factor: float
    ) -> float:
        predicted = _as_probability_vector(beliefs, name="beliefs")
        total = 0.0
        for step, action in enumerate(policy):
            value = self.expected_free_energy(predicted, action)[0]
            total += float(discount_factor**step) * value
            predicted = self.predict_states(predicted, action)
        return float(total)

    def _validate_action(self, action: int) -> int:
        index = int(action)
        if index != action or not 0 <= index < self.num_actions:
            raise ValueError(f"Action out of range: {action}")
        return index
