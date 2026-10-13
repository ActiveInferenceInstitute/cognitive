"""Continuous perception-action loops and discrete control at two timescales."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TypeAlias

import numpy as np
from Things.Continuous_Generic import ContinuousActiveInference

from .models.active_inference.base import ModelState
from .models.active_inference.dispatcher import ActiveInferenceDispatcher, InferenceConfig
from .models.active_inference.generative_model import DiscreteGenerativeModel
from .simulation import Trajectory, corridor_model, run_episode, save_table

JSONValue: TypeAlias = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class ContinuousTrajectory:
    """Continuous states and beliefs include the initial value; other arrays are per step."""

    seed: int
    time: np.ndarray
    states: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    belief_means: np.ndarray
    free_energy: np.ndarray
    config: dict[str, JSONValue]

    def save_json(self, path: str | Path) -> Path:
        """Write the recorded arrays and reproduction parameters as JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            data[descriptor.name] = value.tolist() if isinstance(value, np.ndarray) else value
        data.update(schema_version=1, kind="continuous")
        destination.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return destination

    def save_csv(self, path: str | Path) -> Path:
        """Write time, state, action, observation and free energy for each step."""
        rows = []
        for t, energy in enumerate(self.free_energy):
            row = {"time": self.time[t + 1], "free_energy": energy}
            for name, values in (
                ("state", self.states[t + 1]),
                ("action", self.actions[t]),
                ("observation", self.observations[t]),
            ):
                row.update({f"{name}_{i}": float(value) for i, value in enumerate(values)})
            rows.append(row)
        return save_table(rows, path)


def run_continuous_episode(
    agent: ContinuousActiveInference,
    steps: int,
    *,
    seed: int = 0,
    initial_state: np.ndarray | None = None,
    process_dynamics: np.ndarray | None = None,
    process_noise: float = 0.0,
    sensor_noise: float = 0.0,
    control_gain: float = 1.0,
) -> ContinuousTrajectory:
    """Advance a continuous agent and an Euler-discretized controlled world.

    World dynamics are ``dx/dt = process_dynamics @ x - control_gain * action``;
    process noise has standard deviation ``process_noise * sqrt(dt)``. Each
    observation updates the agent, whose returned residual action controls the
    next world step. The passed agent is advanced from its current state.
    This is the existing finite generalized-coordinate agent, not an ODE solver
    or an expected-free-energy action optimizer.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for name, value in (
        ("process_noise", process_noise),
        ("sensor_noise", sensor_noise),
        ("control_gain", control_gain),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    initial = np.ones(agent.n_states) if initial_state is None else np.asarray(initial_state, float)
    dynamics = (
        -np.eye(agent.n_states) if process_dynamics is None else np.asarray(process_dynamics, float)
    )
    if initial.shape != (agent.n_states,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must be a finite state vector")
    if dynamics.shape != (agent.n_states, agent.n_states) or not np.all(np.isfinite(dynamics)):
        raise ValueError("process_dynamics must be a finite square state matrix")
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    states = np.zeros((steps + 1, agent.n_states))
    means = np.zeros((steps + 1, agent.n_states, agent.n_orders))
    observations = np.zeros((steps, agent.n_obs))
    actions = np.zeros((steps, agent.n_states))
    energy = np.zeros(steps)
    states[0], means[0] = initial, agent.state.belief_means
    start_time = agent.state.time
    config = {
        name: getattr(agent, name)
        for name in (
            "dt",
            "alpha",
            "n_states",
            "n_obs",
            "n_orders",
            "dynamics_weight",
            "precision_learning_rate",
            "precision_min",
            "precision_max",
        )
    }
    config.update(
        observation_matrix=agent.observation_matrix.tolist(),
        dynamics_matrix=agent.dynamics_matrix.tolist(),
        observation_precision=agent.observation_precision.tolist(),
        state_precision=agent.state.belief_precisions[:, 0].tolist(),
        initial_belief_precisions=agent.state.belief_precisions.tolist(),
        process_dynamics=dynamics.tolist(),
        process_noise=process_noise,
        sensor_noise=sensor_noise,
        control_gain=control_gain,
    )
    for t in range(steps):
        previous_action = actions[t - 1] if t else np.zeros(agent.n_states)
        states[t + 1] = states[t] + agent.dt * (
            dynamics @ states[t] - control_gain * previous_action
        )
        states[t + 1] += rng.normal(0, process_noise * np.sqrt(agent.dt), agent.n_states)
        observations[t] = agent.observation_matrix @ states[t + 1]
        observations[t] += rng.normal(0, sensor_noise, agent.n_obs)
        actions[t], energy[t] = agent.step(observations[t])
        means[t + 1] = agent.state.belief_means
        if not all(
            np.all(np.isfinite(a)) for a in (states[t + 1], actions[t], means[t + 1], energy[t])
        ):
            raise ValueError(f"Continuous episode became non-finite at step {t}")
    return ContinuousTrajectory(
        seed,
        start_time + np.arange(steps + 1) * agent.dt,
        states,
        observations,
        actions,
        means,
        energy,
        config,
    )


@dataclass(frozen=True, slots=True)
class Hierarchy:
    """Upper actions select lower log preferences; lower observations inform the upper model."""

    lower: DiscreteGenerativeModel
    upper: DiscreteGenerativeModel
    goal_preferences: np.ndarray
    observation_map: np.ndarray

    def __post_init__(self) -> None:
        goals = np.asarray(self.goal_preferences, float)
        mapping = np.asarray(self.observation_map)
        if goals.shape != (self.upper.num_actions, self.lower.num_observations) or not np.all(
            np.isfinite(goals)
        ):
            raise ValueError("goal_preferences must have shape (upper actions, lower observations)")
        if (
            mapping.shape != (self.lower.num_observations,)
            or mapping.dtype.kind not in "iu"
            or np.any(mapping < 0)
            or np.any(mapping >= self.upper.num_observations)
        ):
            raise ValueError(
                "observation_map must map each lower observation to an upper observation"
            )
        object.__setattr__(self, "goal_preferences", goals.copy())
        object.__setattr__(self, "observation_map", mapping.copy())


def corridor_hierarchy() -> Hierarchy:
    """Two upper goals (left and right) controlling a four-cell corridor."""
    upper = DiscreteGenerativeModel(
        A=np.eye(2),
        B=np.stack(
            [np.array([[1.0, 1.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 1.0]])], axis=2
        ),
        C=np.array([0.0, 2.0]),
        D=np.array([1.0, 0.0]),
        E=np.array([0.5, 0.5]),
    )
    return Hierarchy(
        corridor_model(),
        upper,
        np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]]),
        np.array([0, 0, 1, 1]),
    )


@dataclass(frozen=True, slots=True)
class HierarchicalTrajectory:
    """Lower episode plus upper decisions and beliefs at each decision boundary."""

    lower: Trajectory
    decision_steps: np.ndarray
    goals: np.ndarray
    upper_beliefs: np.ndarray
    upper_posteriors: np.ndarray
    upper_config: dict[str, JSONValue]
    goal_preferences: np.ndarray
    observation_map: np.ndarray
    upper_model: dict[str, JSONValue]

    def save_json(self, path: str | Path) -> Path:
        """Write both levels, their mapping, preferences and upper model."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for descriptor in fields(self):
            if descriptor.name != "lower":
                value = getattr(self, descriptor.name)
                data[descriptor.name] = value.tolist() if isinstance(value, np.ndarray) else value
        data.update(schema_version=1, kind="hierarchical", lower=self.lower.to_dict())
        destination.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return destination


def run_hierarchical_episode(
    hierarchy: Hierarchy,
    lower_config: InferenceConfig,
    upper_config: InferenceConfig,
    steps: int,
    *,
    decision_interval: int = 5,
    seed: int = 0,
) -> HierarchicalTrajectory:
    """Choose an upper goal every interval and execute lower perception-action cycles.

    The upper posterior is updated from the last lower observation in each
    interval. Lower beliefs and physical state persist across boundaries.
    Separate child seeds isolate upper sampling from the lower episodes.
    """
    if (
        type(steps) is not int
        or steps < 1
        or type(decision_interval) is not int
        or decision_interval < 1
    ):
        raise ValueError("steps and decision_interval must be positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    decisions = np.arange(0, steps, decision_interval)
    upper_seed, lower_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(upper_seed)
    children = lower_seed.spawn(len(decisions))
    dispatcher = ActiveInferenceDispatcher(replace(upper_config), hierarchy.upper, rng=rng)
    beliefs = np.zeros((len(decisions) + 1, hierarchy.upper.num_states))
    beliefs[0] = hierarchy.upper.D
    goals = np.zeros(len(decisions), dtype=int)
    posteriors = np.zeros((len(decisions), hierarchy.upper.num_actions))
    chunks: list[Trajectory] = []
    for i, start in enumerate(decisions):
        state = ModelState(beliefs[i], hierarchy.upper.E, upper_config.precision_init, 0.0, 0.0)
        posteriors[i] = dispatcher.dispatch_policy_inference(state)
        goals[i] = rng.choice(hierarchy.upper.num_actions, p=posteriors[i])
        prior = chunks[-1].beliefs[-1] if chunks else hierarchy.lower.D
        low_model = replace(hierarchy.lower, C=hierarchy.goal_preferences[goals[i]], D=prior)
        chunk = run_episode(
            low_model,
            lower_config,
            min(decision_interval, steps - int(start)),
            seed=int(children[i].generate_state(1)[0]),
            process=hierarchy.lower,
            initial_state=int(chunks[-1].states[-1]) if chunks else None,
        )
        chunks.append(chunk)
        state.beliefs = hierarchy.upper.predict_states(beliefs[i], int(goals[i]))
        observation = int(hierarchy.observation_map[chunk.observations[-1]])
        beliefs[i + 1] = dispatcher.dispatch_belief_update(observation, state)
    arrays = {}
    for name in (
        "beliefs",
        "states",
        "observations",
        "actions",
        "policy_posteriors",
        "efe",
        "risk",
        "ambiguity",
        "epistemic",
        "free_energy",
        "complexity",
        "accuracy",
    ):
        arrays[name] = np.concatenate(
            [
                getattr(chunk, name)[1:]
                if i and name in ("beliefs", "states")
                else getattr(chunk, name)
                for i, chunk in enumerate(chunks)
            ]
        )
    lower = replace(
        chunks[0], seed=seed, preferred_observation=int(np.argmax(hierarchy.lower.C)), **arrays
    )
    upper_record = {key: value.tolist() for key, value in hierarchy.upper.__dict__.items()}
    return HierarchicalTrajectory(
        lower,
        decisions,
        goals,
        beliefs,
        posteriors,
        {**upper_config.__dict__, "decision_interval": decision_interval},
        hierarchy.goal_preferences,
        hierarchy.observation_map,
        upper_record,
    )
