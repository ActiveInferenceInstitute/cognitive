from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class ContinuousState:
    belief_means: np.ndarray
    belief_precisions: np.ndarray
    time: float = 0.0


class ContinuousActiveInference:
    def __init__(
        self,
        n_states: int = 2,
        n_obs: int = 2,
        n_orders: int = 3,
        dt: float = 0.01,
        alpha: float = 0.1,
    ):
        self.n_states = n_states
        self.n_obs = n_obs
        self.n_orders = n_orders
        self.dt = dt
        self.alpha = alpha
        self.state = ContinuousState(
            belief_means=np.zeros((n_states, n_orders), dtype=float),
            belief_precisions=np.ones((n_states, n_orders), dtype=float),
        )

    def _sensory_mapping(self, belief_means: np.ndarray) -> np.ndarray:
        return np.asarray(belief_means[: self.n_obs, 0], dtype=float)

    def _compute_free_energy(self, observation: np.ndarray, belief_means: np.ndarray) -> float:
        prediction = self._sensory_mapping(belief_means)
        obs_error = np.asarray(observation, dtype=float) - prediction
        dynamic_error = np.diff(belief_means, axis=1)
        accuracy = 0.5 * float(np.sum(obs_error**2))
        complexity = 0.5 * self.alpha * float(np.sum(dynamic_error**2))
        return accuracy + complexity

    def step(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        observation = np.asarray(observation, dtype=float)
        prediction = self._sensory_mapping(self.state.belief_means)
        prediction_error = observation - prediction

        self.state.belief_means[: self.n_obs, 0] += self.alpha * prediction_error
        if self.n_orders > 1:
            self.state.belief_means[:, 0] += self.dt * self.state.belief_means[:, 1]
        if self.n_orders > 2:
            self.state.belief_means[:, 1] += self.dt * self.state.belief_means[:, 2]
            self.state.belief_means[:, 2] += -self.dt * self.state.belief_means[:, 0]

        self.state.belief_precisions = np.maximum(
            1.0 / (np.abs(self.state.belief_means) + 1.0), EPS
        )
        free_energy = self._compute_free_energy(observation, self.state.belief_means)
        action = -prediction_error
        if action.size < self.n_states:
            action = np.pad(action, (0, self.n_states - action.size))
        self.state.time += self.dt
        return action[: self.n_states], float(free_energy)
