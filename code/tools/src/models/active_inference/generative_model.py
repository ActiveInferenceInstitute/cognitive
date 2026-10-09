"""Validated discrete generative models used by Active Inference inference."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice, product

import numpy as np

from ...utils.matrix_utils import as_integer

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
    """Return Shannon entropy in nats, exactly zero for a deterministic input.

    Zero-probability outcomes contribute nothing, so the result cannot go
    slightly negative the way a uniform ``log(p + EPS)`` floor allows.
    """
    probabilities = _normalize(distribution)
    terms = np.where(probabilities > 0, probabilities * np.log(np.maximum(probabilities, EPS)), 0.0)
    return float(-np.sum(terms))


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
        """Return ``P(s | o)`` for one observation index or an observation distribution.

        An integer ``observation`` selects a row of ``A``; a vector is treated as
        a distribution over observations and weights the likelihood by
        ``A.T @ observation``. Fractional indices are rejected rather than
        truncated. If the observation has exactly zero probability under every
        state the prior supports, no update is possible and the prior is
        returned unchanged. Evidence is compared with zero, not with a
        tolerance: normalizing is scale-invariant, so a small but positive
        likelihood still yields an exact posterior.
        """
        prior_vector = _as_probability_vector(prior, name="prior")
        if prior_vector.shape != self.D.shape:
            raise ValueError(f"prior must have shape {self.D.shape}")
        if np.ndim(observation) == 0:
            index = as_integer(observation, "Observation")
            if not 0 <= index < self.num_observations:
                raise ValueError(f"Observation out of range: {index}")
            likelihood = self.A[index]
        else:
            observation_vector = _as_probability_vector(observation, name="observation")
            if observation_vector.shape != (self.num_observations,):
                raise ValueError(f"observation must have shape {(self.num_observations,)}")
            likelihood = self.A.T @ observation_vector
        posterior = prior_vector * np.maximum(likelihood, 0.0)
        evidence = float(posterior.sum())
        if evidence <= 0.0:
            return prior_vector
        return posterior / evidence

    def expected_free_energy(
        self, beliefs: np.ndarray, action: int
    ) -> tuple[float, float, float, float]:
        """Return total EFE, risk, ambiguity, and epistemic information gain.

        The total is the canonical risk-plus-ambiguity decomposition

        .. math::
            G = \\underbrace{D_{KL}[q(o) \\| p^*(o)]}_{\\text{risk}}
              + \\underbrace{\\mathbb{E}_{q(s)}[H[P(o \\mid s)]]}_{\\text{ambiguity}},

        which is algebraically identical to expected surprise minus expected
        information gain, because ``ambiguity == H[q(o)] - I(s; o)``.  The
        epistemic gain ``I(s; o)`` is therefore already accounted for inside
        ``ambiguity`` and is returned only as a diagnostic component; adding it
        to the total a second time would weight information seeking twice.
        ``test_efe_decompositions_agree`` pins this identity.
        """
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
        total = risk + ambiguity
        return float(total), float(risk), float(ambiguity), float(epistemic)

    def enumerate_policies(self, horizon: int, limit: int | None = None) -> list[tuple[int, ...]]:
        """Enumerate action sequences of length ``horizon`` in lexicographic order.

        ``limit`` truncates lazily, so a small limit over a large action-horizon
        product never materializes the full ``num_actions ** horizon`` space.
        """
        if horizon < 1:
            raise ValueError("horizon must be at least one")
        if limit is not None and limit < 1:
            raise ValueError("policy limit must be positive")
        policies = product(range(self.num_actions), repeat=horizon)
        return list(policies if limit is None else islice(policies, limit))

    def evaluate_policy(
        self,
        beliefs: np.ndarray,
        policy: tuple[int, ...],
        discount_factor: float,
        epistemic_weight: float = 0.0,
    ) -> float:
        """Return the discounted expected free energy of one action sequence.

        ``epistemic_weight`` subtracts an *additional* multiple of the expected
        information gain from every step, on top of the epistemic drive that the
        canonical objective already contains through ``ambiguity``.  The default
        ``0.0`` is the canonical objective; positive values bias the agent toward
        information seeking, and ``1.0`` reproduces the objective this package
        used before version 1.1.0.
        """
        if not np.isfinite(epistemic_weight) or epistemic_weight < 0:
            raise ValueError("epistemic_weight must be finite and non-negative")
        try:
            valid_discount = bool(np.isfinite(discount_factor) and 0.0 < discount_factor <= 1.0)
        except TypeError:
            valid_discount = False
        if not valid_discount:
            raise ValueError("discount_factor must be in (0, 1]")
        policy = tuple(policy)
        if len(policy) == 0:
            raise ValueError("policy must contain at least one action")
        predicted = _as_probability_vector(beliefs, name="beliefs")
        total = 0.0
        for step, action in enumerate(policy):
            value, _risk, _ambiguity, epistemic = self.expected_free_energy(predicted, action)
            total += float(discount_factor**step) * (value - epistemic_weight * epistemic)
            predicted = self.predict_states(predicted, action)
        return float(total)

    def _validate_action(self, action: int) -> int:
        index = int(action)
        if index != action or not 0 <= index < self.num_actions:
            raise ValueError(f"Action out of range: {action}")
        return index
