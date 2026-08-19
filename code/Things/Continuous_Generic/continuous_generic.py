"""Validated continuous-time active inference primitives.

The implementation uses a finite generalized-coordinate approximation.  The
zeroth coordinate is the latent state, higher coordinates are its successive
derivatives, and the configured dynamics matrix describes the expected
derivative at every order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def _finite_array(value: np.ndarray | list[float], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain at least one finite value")
    return array


@dataclass
class ContinuousState:
    """Mutable state of a continuous agent."""

    belief_means: np.ndarray
    belief_precisions: np.ndarray
    time: float = 0.0

    def copy(self) -> ContinuousState:
        return ContinuousState(self.belief_means.copy(), self.belief_precisions.copy(), self.time)


class ContinuousActiveInference:
    """Perform precision-weighted generalized-coordinate updates.

    Parameters are intentionally explicit so a run can be reproduced from a
    configuration file.  ``observation_matrix`` maps latent states to
    observations and ``dynamics_matrix`` maps each generalized coordinate to
    its expected derivative.  Rectangular observation matrices are supported.

    ``alpha`` is the gradient-descent step size and ``dynamics_weight`` is the
    weight of the dynamic (generalized-coordinate) term in the free energy.
    ``step`` applies exact gradient descent on the free energy that
    ``_compute_free_energy`` reports, so a finite-difference gradient of that
    function reproduces the applied update; the regression test
    ``test_step_follows_free_energy_gradient`` pins the identity for
    non-identity dynamics matrices.
    """

    def __init__(
        self,
        n_states: int = 2,
        n_obs: int = 2,
        n_orders: int = 3,
        dt: float = 0.01,
        alpha: float = 0.1,
        observation_matrix: np.ndarray | list[list[float]] | None = None,
        dynamics_matrix: np.ndarray | list[list[float]] | None = None,
        observation_precision: float | np.ndarray = 1.0,
        state_precision: float | np.ndarray = 1.0,
        precision_learning_rate: float = 0.0,
        dynamics_weight: float = 1.0,
        seed: int | None = 0,
    ) -> None:
        self.n_states = self._positive_int(n_states, "n_states")
        self.n_obs = self._positive_int(n_obs, "n_obs")
        self.n_orders = self._positive_int(n_orders, "n_orders")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not 0.0 <= precision_learning_rate <= 1.0:
            raise ValueError("precision_learning_rate must be in [0, 1]")
        if not np.isfinite(dynamics_weight) or dynamics_weight < 0:
            raise ValueError("dynamics_weight must be finite and non-negative")
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a non-negative integer or None")
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.dynamics_weight = float(dynamics_weight)
        self.precision_learning_rate = float(precision_learning_rate)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        default_observation = np.zeros((self.n_obs, self.n_states), dtype=float)
        default_observation[:, : min(self.n_obs, self.n_states)] = np.eye(
            min(self.n_obs, self.n_states)
        )
        self.observation_matrix = self._validate_matrix(
            observation_matrix if observation_matrix is not None else default_observation,
            (self.n_obs, self.n_states),
            "observation_matrix",
        )
        default_dynamics = np.eye(self.n_states, dtype=float)
        self.dynamics_matrix = self._validate_matrix(
            dynamics_matrix if dynamics_matrix is not None else default_dynamics,
            (self.n_states, self.n_states),
            "dynamics_matrix",
        )
        self.observation_precision = self._precision_vector(
            observation_precision, self.n_obs, "observation_precision"
        )
        state_precision_vector = self._precision_vector(
            state_precision, self.n_states, "state_precision"
        )
        self._initial_state_precision = np.repeat(
            state_precision_vector[:, None], self.n_orders, axis=1
        )
        self.state = ContinuousState(
            belief_means=np.zeros((self.n_states, self.n_orders), dtype=float),
            belief_precisions=self._initial_state_precision.copy(),
        )

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _validate_matrix(
        value: np.ndarray | list[list[float]], shape: tuple[int, int], name: str
    ) -> np.ndarray:
        matrix = _finite_array(value, name=name)
        if matrix.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
        return matrix.copy()

    @staticmethod
    def _precision_vector(value: float | np.ndarray, size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(size, float(array))
        if array.shape != (size,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
            raise ValueError(f"{name} must be a positive scalar or shape {(size,)}")
        return array.copy()

    def reset(self, belief_means: np.ndarray | None = None) -> None:
        """Reset time and beliefs, optionally to a validated initial value."""
        if belief_means is None:
            means = np.zeros((self.n_states, self.n_orders), dtype=float)
        else:
            means = _finite_array(belief_means, name="belief_means")
            if means.shape != (self.n_states, self.n_orders):
                raise ValueError(f"belief_means must have shape {(self.n_states, self.n_orders)}")
            means = means.copy()
        self.state = ContinuousState(means, self._initial_state_precision.copy())

    def _sensory_mapping(self, belief_means: np.ndarray) -> np.ndarray:
        means = _finite_array(belief_means, name="belief_means")
        if means.shape != (self.n_states, self.n_orders):
            raise ValueError(f"belief_means must have shape {(self.n_states, self.n_orders)}")
        return self.observation_matrix @ means[:, 0]

    def _compute_free_energy(self, observation: np.ndarray, belief_means: np.ndarray) -> float:
        obs = _finite_array(observation, name="observation").reshape(-1)
        if obs.shape != (self.n_obs,):
            raise ValueError(f"observation must have shape {(self.n_obs,)}")
        means = _finite_array(belief_means, name="belief_means")
        prediction_error = obs - self._sensory_mapping(means)
        sensory_energy = 0.5 * float(np.sum(self.observation_precision * prediction_error**2))
        dynamic_energy = 0.0
        for order in range(self.n_orders - 1):
            error = means[:, order + 1] - self.dynamics_matrix @ means[:, order]
            dynamic_energy += 0.5 * float(np.sum(self.state.belief_precisions[:, order] * error**2))
        return sensory_energy + self.dynamics_weight * dynamic_energy

    def step(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        """Update beliefs from one observation and return action and free energy."""
        obs = _finite_array(observation, name="observation").reshape(-1)
        if obs.shape != (self.n_obs,):
            raise ValueError(f"observation must have shape {(self.n_obs,)}")
        means = self.state.belief_means
        prediction_error = obs - self._sensory_mapping(means)
        sensory_gradient = self.observation_matrix.T @ (
            self.observation_precision * prediction_error
        )

        # updates is -dF/dmu for the free energy defined in _compute_free_energy,
        # scaled by the step size alpha. The dynamic term contributes
        # +f' D^T Pi e to the coordinate it predicts from and -f' Pi e to the
        # coordinate it predicts; omitting the D^T Jacobian factor is only
        # correct when dynamics_matrix is the identity.
        updates = np.zeros_like(means)
        updates[:, 0] = sensory_gradient
        for order in range(self.n_orders - 1):
            dynamic_error = means[:, order + 1] - self.dynamics_matrix @ means[:, order]
            weighted = (
                self.dynamics_weight * self.state.belief_precisions[:, order] * dynamic_error
            )
            updates[:, order] += self.dynamics_matrix.T @ weighted
            updates[:, order + 1] -= weighted
        means += self.dt * self.alpha * updates

        if self.precision_learning_rate:
            residual = np.abs(prediction_error)
            observed_precision = 1.0 / np.maximum(residual**2, EPS)
            target = np.full(self.n_obs, observed_precision.mean())
            self.observation_precision = (
                1.0 - self.precision_learning_rate
            ) * self.observation_precision + self.precision_learning_rate * target
        self.state.belief_precisions = np.maximum(self.state.belief_precisions, EPS)
        free_energy = self._compute_free_energy(obs, means)
        action = self.observation_matrix.T @ (self.observation_precision * prediction_error)
        self.state.time += self.dt
        return np.asarray(action, dtype=float), float(free_energy)
