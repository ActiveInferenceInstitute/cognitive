"""Run discrete Active Inference episodes and parameter sweeps.

An *episode* closes the loop the rest of the package leaves open: a hidden
world state evolves under the agent's actions and emits observations, and the
agent updates its beliefs and chooses the next action by minimizing expected
free energy through an :class:`ActiveInferenceDispatcher`.

Per step ``t``, in this order:

1. the dispatcher turns the current beliefs into a distribution over actions;
2. an action is sampled from it;
3. the world transitions with ``process.B`` and emits an observation from
   ``process.A``;
4. the agent predicts its next states with its own ``model.B`` and updates its
   beliefs on the observation.

The world (``process``) defaults to the agent's own model, which is the
standard case of an agent with a correct model. Passing a different
``process`` studies a mismatch.

The world and the agent draw from separate random streams spawned from one
seed, so changing how the agent samples never changes the world's noise and an
episode is exactly reproducible from ``(model, config, steps, seed)``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .models.active_inference.base import ModelState
from .models.active_inference.dispatcher import ActiveInferenceDispatcher, InferenceConfig
from .models.active_inference.generative_model import DiscreteGenerativeModel
from .utils.matrix_utils import kl_divergence

# InferenceConfig fields a sweep may vary, with the type each value must have.
SWEEPABLE = {
    "temperature": float,
    "learning_rate": float,
    "precision_init": float,
    "discount_factor": float,
    "exploration_weight": float,
    "temporal_horizon": int,
    "policy_aggregation": str,
    "method": str,
}

TRAJECTORY_SCHEMA_VERSION = 1


def example_model(
    sensor_noise: float = 0.1, preference: float = 1.0, stickiness: float = 0.9
) -> DiscreteGenerativeModel:
    """A two-state, two-action model with a tunable sensor and a preferred observation.

    ``A`` reports the true state with error rate ``sensor_noise``; action 0
    mostly keeps the state (with probability ``stickiness``) and action 1
    mostly flips it; the agent prefers observation 1 by ``preference`` nats.
    """
    if not 0.0 <= sensor_noise <= 0.5:
        raise ValueError("sensor_noise must be in [0, 0.5]")
    if not 0.5 <= stickiness <= 1.0:
        raise ValueError("stickiness must be in [0.5, 1]")
    keep = np.array([[stickiness, 1 - stickiness], [1 - stickiness, stickiness]])
    flip = np.array([[1 - stickiness, stickiness], [stickiness, 1 - stickiness]])
    return DiscreteGenerativeModel(
        A=np.array([[1 - sensor_noise, sensor_noise], [sensor_noise, 1 - sensor_noise]]),
        B=np.stack([keep, flip], axis=2),
        C=np.array([0.0, float(preference)]),
        D=np.array([0.5, 0.5]),
        E=np.array([0.5, 0.5]),
    )


def corridor_model(
    length: int = 4, slip: float = 0.1, sensor_noise: float = 0.1, preference: float = 2.0
) -> DiscreteGenerativeModel:
    """A corridor of ``length`` cells with a goal in the last one.

    The hidden state is the agent's cell and the observation is a noisy report
    of it. Actions are ``left``, ``stay`` and ``right``; a move succeeds with
    probability ``1 - slip`` and otherwise leaves the agent where it is, and
    walls stop it. The agent prefers to observe the last cell by ``preference``
    nats and starts at the first cell. Unlike :func:`example_model` the actions
    are not symmetric, so planning horizons and the way policies are combined
    change what the agent does.
    """
    if length < 2:
        raise ValueError("length must be at least two")
    if not 0.0 <= slip < 1.0:
        raise ValueError("slip must be in [0, 1)")
    if not 0.0 <= sensor_noise < 1.0 - 1.0 / length:
        raise ValueError("sensor_noise must be below 1 - 1/length so the sensor beats chance")
    transitions = np.zeros((length, length, 3))
    for cell in range(length):
        for action, step in enumerate((-1, 0, 1)):
            target = min(max(cell + step, 0), length - 1)
            transitions[target, cell, action] += 1.0 - slip
            transitions[cell, cell, action] += slip
    sensor = np.full((length, length), sensor_noise / (length - 1))
    np.fill_diagonal(sensor, 1.0 - sensor_noise)
    preferences = np.zeros(length)
    preferences[-1] = float(preference)
    start = np.zeros(length)
    start[0] = 1.0
    return DiscreteGenerativeModel(
        A=sensor, B=transitions, C=preferences, D=start, E=np.full(3, 1 / 3)
    )


def load_model(path: str | Path) -> tuple[DiscreteGenerativeModel, dict[str, list[str]]]:
    """Read a generative model and optional labels from a YAML file.

    The file holds ``A`` (observations x states), one of ``B`` (states x
    states x actions, the package convention) or ``B_by_action`` (a list with
    one states x states matrix per action, easier to write by hand), ``C``
    (log preferences), ``D`` and ``E``, and may hold ``labels`` with lists
    named ``states``, ``actions`` and ``observations``.
    """
    import yaml

    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(f"Cannot read model file {source}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"{source} is not valid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must contain a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{source}: model keys must be strings")
    allowed = {"A", "B", "B_by_action", "C", "D", "E", "labels", "description"}
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise ValueError(f"{source}: unknown key(s) {unknown}; allowed are {sorted(allowed)}")
    if ("B" in raw) == ("B_by_action" in raw):
        raise ValueError(f"{source} must define exactly one of B and B_by_action")
    matrices = dict(raw)
    if "B_by_action" in matrices:
        if not isinstance(matrices["B_by_action"], list) or not matrices["B_by_action"]:
            raise ValueError(f"{source}: B_by_action must be a non-empty list of matrices")
        matrices["B"] = np.stack(
            [np.asarray(matrix, dtype=float) for matrix in matrices.pop("B_by_action")], axis=2
        )
    labels = raw.get("labels", {})
    if not isinstance(labels, dict) or set(labels).difference(
        {"states", "actions", "observations"}
    ):
        raise ValueError(f"{source}: labels must map states, actions, observations to lists")
    model = DiscreteGenerativeModel.from_config({k: matrices[k] for k in "ABCDE" if k in matrices})
    for key, count in (
        ("states", model.num_states),
        ("actions", model.num_actions),
        ("observations", model.num_observations),
    ):
        if key in labels and (
            not isinstance(labels[key], list)
            or len(labels[key]) != count
            or not all(isinstance(item, str) for item in labels[key])
        ):
            raise ValueError(f"{source}: {key} labels must contain {count} strings")
    return model, labels


@dataclass(frozen=True)
class Trajectory:
    """Everything recorded during one episode.

    Arrays indexed by step have length ``steps``; ``beliefs`` and ``states``
    have length ``steps + 1`` because they include the initial value.
    ``efe`` and its parts are ``(steps, actions)``: the expected free energy
    of each action *at the beliefs held when that action was chosen*.
    """

    seed: int
    config: Mapping[str, Any]
    beliefs: np.ndarray
    states: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    policy_posteriors: np.ndarray
    efe: np.ndarray
    risk: np.ndarray
    ambiguity: np.ndarray
    epistemic: np.ndarray
    free_energy: np.ndarray
    complexity: np.ndarray
    accuracy: np.ndarray
    preferred_observation: int
    state_labels: tuple[str, ...] = field(default_factory=tuple)
    action_labels: tuple[str, ...] = field(default_factory=tuple)
    observation_labels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def steps(self) -> int:
        return int(self.actions.shape[0])

    @property
    def belief_accuracy(self) -> float:
        """Mean probability the agent assigned to the true state after each update."""
        taken = self.beliefs[1:][np.arange(self.steps), self.states[1:]]
        return float(np.mean(taken))

    @property
    def preference_rate(self) -> float:
        """Fraction of observations equal to the agent's most preferred one."""
        return float(np.mean(self.observations == self.preferred_observation))

    def summary(self) -> dict[str, Any]:
        """Scalar statistics of the episode, JSON-serializable."""
        entropy = -np.sum(
            self.policy_posteriors * np.log(np.maximum(self.policy_posteriors, 1e-12)), axis=1
        )
        return {
            "steps": self.steps,
            "seed": self.seed,
            "mean_free_energy": float(np.mean(self.free_energy)),
            "mean_complexity": float(np.mean(self.complexity)),
            "mean_accuracy": float(np.mean(self.accuracy)),
            "belief_accuracy": self.belief_accuracy,
            "preference_rate": self.preference_rate,
            "mean_policy_entropy": float(np.mean(entropy)),
            "action_counts": np.bincount(
                self.actions, minlength=self.policy_posteriors.shape[1]
            ).tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable view; arrays become nested lists."""
        return {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "seed": self.seed,
            "config": dict(self.config),
            "labels": {
                "states": list(self.state_labels),
                "actions": list(self.action_labels),
                "observations": list(self.observation_labels),
            },
            "preferred_observation": self.preferred_observation,
            "summary": self.summary(),
            "beliefs": self.beliefs.tolist(),
            "states": self.states.tolist(),
            "observations": self.observations.tolist(),
            "actions": self.actions.tolist(),
            "policy_posteriors": self.policy_posteriors.tolist(),
            "efe": self.efe.tolist(),
            "risk": self.risk.tolist(),
            "ambiguity": self.ambiguity.tolist(),
            "epistemic": self.epistemic.tolist(),
            "free_energy": self.free_energy.tolist(),
            "complexity": self.complexity.tolist(),
            "accuracy": self.accuracy.tolist(),
        }

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Trajectory:
        """Parse schema 1 data, rejecting malformed arrays and inconsistent records.

        The saved summary is derived data; it is recomputed from the arrays.
        """
        if raw.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError("Unsupported trajectory schema_version")
        if type(raw["schema_version"]) is not int:
            raise ValueError("schema_version must be an integer")
        required = {
            "seed",
            "config",
            "labels",
            "preferred_observation",
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
        }
        if required.difference(raw):
            raise ValueError(f"Missing trajectory fields: {sorted(required.difference(raw))}")
        if set(raw).difference(required | {"schema_version", "summary"}):
            raise ValueError("Unknown trajectory fields")
        for key in ("seed", "preferred_observation"):
            if type(raw[key]) is not int or raw[key] < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        if not isinstance(raw["config"], dict) or not isinstance(raw["labels"], dict):
            raise ValueError("config and labels must be mappings")
        try:
            json.dumps(raw["config"], allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("config must contain finite JSON values") from error
        arrays = {}
        for name in required.difference({"seed", "config", "labels", "preferred_observation"}):
            try:
                entries = np.asarray(raw[name], dtype=object)
                if any(isinstance(item, bool) for item in entries.flat):
                    raise ValueError(f"{name} must not contain booleans")
                candidate = np.asarray(raw[name])
                if candidate.dtype.kind not in "iuf":
                    raise ValueError(f"{name} must be a numeric array")
                arrays[name] = candidate.astype(float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a numeric array") from error
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"{name} must contain finite values")
        beliefs, policies = arrays["beliefs"], arrays["policy_posteriors"]
        if policies.ndim != 2 or min(policies.shape) < 1:
            raise ValueError("policy_posteriors must have shape (steps, actions)")
        steps, actions = policies.shape
        if beliefs.ndim != 2 or beliefs.shape[0] != steps + 1 or beliefs.shape[1] < 1:
            raise ValueError("beliefs must have shape (steps + 1, states)")
        for name in ("beliefs", "policy_posteriors"):
            if np.any(arrays[name] < 0) or not np.allclose(arrays[name].sum(axis=1), 1):
                raise ValueError(f"{name} must contain probability rows")
        for name in ("efe", "risk", "ambiguity", "epistemic"):
            if arrays[name].shape != (steps, actions):
                raise ValueError(f"{name} must have shape (steps, actions)")
        for name in ("free_energy", "complexity", "accuracy", "observations", "actions", "states"):
            if arrays[name].shape != (steps + (name == "states"),):
                raise ValueError(f"{name} has inconsistent length")
        for name, limit in (
            ("states", beliefs.shape[1]),
            ("actions", actions),
            ("observations", None),
        ):
            array = np.asarray(raw[name])
            if np.any(array < 0) or np.any(array != np.floor(array)):
                raise ValueError(f"{name} must contain non-negative integers")
            if int(array.max()) > np.iinfo(np.int64).max:
                raise ValueError(f"{name} index exceeds the supported integer range")
            if limit is not None and np.any(array >= limit):
                raise ValueError(f"{name} index out of range")
            arrays[name] = array.astype(np.int64)
        if not np.allclose(arrays["free_energy"], arrays["complexity"] - arrays["accuracy"]):
            raise ValueError("free_energy must equal complexity - accuracy")
        if not np.allclose(arrays["efe"], arrays["risk"] + arrays["ambiguity"]):
            raise ValueError("efe must equal risk + ambiguity")
        if np.any(arrays["complexity"] < -1e-8) or np.any(arrays["epistemic"] < -1e-8):
            raise ValueError("complexity and epistemic must be non-negative")
        labels = raw["labels"]
        if set(labels).difference({"states", "actions", "observations"}):
            raise ValueError("Unknown trajectory labels")
        for name, count in (
            ("states", beliefs.shape[1]),
            ("actions", actions),
            ("observations", None),
        ):
            given = labels.get(name, [])
            if not isinstance(given, list) or not all(isinstance(item, str) for item in given):
                raise ValueError(f"{name} labels must be a list of strings")
            if given and count is not None and len(given) != count:
                raise ValueError(f"{name} labels do not match dimensions")
        obs_labels = labels.get("observations", [])
        if obs_labels and (
            arrays["observations"].max() >= len(obs_labels)
            or raw["preferred_observation"] >= len(obs_labels)
        ):
            raise ValueError("observation index out of range")
        return cls(
            seed=raw["seed"],
            config=dict(raw["config"]),
            **arrays,
            preferred_observation=raw["preferred_observation"],
            state_labels=tuple(labels.get("states", [])),
            action_labels=tuple(labels.get("actions", [])),
            observation_labels=tuple(obs_labels),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> Trajectory:
        """Load a validated discrete trajectory from UTF-8 JSON."""
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ValueError(f"Invalid trajectory JSON: {error}") from error
        if not isinstance(raw, dict):
            raise ValueError("Trajectory JSON must contain a mapping")
        return cls.from_dict(raw)

    def save_csv(self, path: str | Path) -> Path:
        """One row per step: what happened and the agent's free energy."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["step", "state", "action", "observation", "free_energy", "complexity", "accuracy"]
            )
            for step in range(self.steps):
                writer.writerow(
                    [
                        step,
                        int(self.states[step + 1]),
                        int(self.actions[step]),
                        int(self.observations[step]),
                        f"{self.free_energy[step]:.12g}",
                        f"{self.complexity[step]:.12g}",
                        f"{self.accuracy[step]:.12g}",
                    ]
                )
        return destination


def _sample(rng: np.random.Generator, distribution: np.ndarray) -> int:
    return int(rng.choice(distribution.shape[0], p=distribution))


def run_episode(
    model: DiscreteGenerativeModel,
    config: InferenceConfig,
    steps: int,
    *,
    seed: int | None = None,
    process: DiscreteGenerativeModel | None = None,
    initial_state: int | None = None,
    labels: Mapping[str, Sequence[str]] | None = None,
) -> Trajectory:
    """Simulate ``steps`` perception-action cycles and record them.

    ``seed`` defaults to ``config.seed`` (``0`` when that is ``None``).
    ``process`` is the real world and must have the same dimensions as
    ``model``. ``initial_state`` fixes the world's starting state; by default
    it is drawn from ``process.D``.
    """
    if steps < 1:
        raise ValueError("steps must be at least one")
    world = process or model
    if (world.num_states, world.num_observations, world.num_actions) != (
        model.num_states,
        model.num_observations,
        model.num_actions,
    ):
        raise ValueError(
            "process and model must have the same numbers of states, observations, actions"
        )
    if initial_state is not None and not 0 <= initial_state < world.num_states:
        raise ValueError(f"initial_state out of range: {initial_state}")
    episode_seed = int(config.seed or 0) if seed is None else int(seed)
    world_seq, agent_seq = np.random.SeedSequence(episode_seed).spawn(2)
    world_rng, agent_rng = np.random.default_rng(world_seq), np.random.default_rng(agent_seq)
    dispatcher = ActiveInferenceDispatcher(replace(config), model, rng=agent_rng)

    states = np.zeros(steps + 1, dtype=int)
    states[0] = _sample(world_rng, world.D) if initial_state is None else initial_state
    beliefs = np.zeros((steps + 1, model.num_states))
    beliefs[0] = model.D
    n_actions = model.num_actions
    shape = (steps, n_actions)
    observations, actions = np.zeros(steps, dtype=int), np.zeros(steps, dtype=int)
    posteriors = np.zeros(shape)
    efe, risk = np.zeros(shape), np.zeros(shape)
    ambiguity, epistemic = np.zeros(shape), np.zeros(shape)
    free_energy, complexity, accuracy = (np.zeros(steps) for _ in range(3))

    for step in range(steps):
        held = ModelState(
            beliefs=beliefs[step].copy(),
            policies=model.E.copy(),
            precision=dispatcher.config.precision_init,
            free_energy=0.0,
            prediction_error=0.0,
        )
        posteriors[step] = dispatcher.dispatch_policy_inference(held)
        for action in range(n_actions):
            (
                efe[step, action],
                risk[step, action],
                ambiguity[step, action],
                epistemic[step, action],
            ) = model.expected_free_energy(beliefs[step], action)
        actions[step] = _sample(agent_rng, posteriors[step])

        states[step + 1] = _sample(world_rng, world.B[:, states[step], actions[step]])
        observations[step] = _sample(world_rng, world.A[:, states[step + 1]])

        predicted = model.predict_states(beliefs[step], int(actions[step]))
        updated = dispatcher.dispatch_belief_update(
            int(observations[step]), _with_beliefs(held, predicted)
        )
        beliefs[step + 1] = updated
        complexity[step] = kl_divergence(updated, predicted)
        accuracy[step] = float(
            np.sum(updated * np.log(np.maximum(model.A[observations[step]], 1e-12)))
        )
        free_energy[step] = complexity[step] - accuracy[step]

    label_map = labels or {}
    return Trajectory(
        seed=episode_seed,
        config=_config_record(dispatcher.config),
        beliefs=beliefs,
        states=states,
        observations=observations,
        actions=actions,
        policy_posteriors=posteriors,
        efe=efe,
        risk=risk,
        ambiguity=ambiguity,
        epistemic=epistemic,
        free_energy=free_energy,
        complexity=complexity,
        accuracy=accuracy,
        preferred_observation=int(np.argmax(model.C)),
        state_labels=tuple(label_map.get("states", ())),
        action_labels=tuple(label_map.get("actions", ())),
        observation_labels=tuple(label_map.get("observations", ())),
    )


def _with_beliefs(state: ModelState, beliefs: np.ndarray) -> ModelState:
    """A copy of ``state`` holding ``beliefs``."""
    copy = state.copy()
    copy.beliefs = np.asarray(beliefs, dtype=float).copy()
    return copy


def _config_record(config: InferenceConfig) -> dict[str, Any]:
    return {
        "method": config.method.value,
        "policy_aggregation": config.policy_aggregation.value,
        "temporal_horizon": config.temporal_horizon,
        "learning_rate": config.learning_rate,
        "precision_init": config.precision_init,
        "temperature": config.temperature,
        "discount_factor": config.discount_factor,
        "exploration_weight": config.exploration_weight,
        "policy_limit": config.policy_limit,
        "num_samples": config.num_samples,
    }


@dataclass(frozen=True)
class SweepRow:
    """One episode of a sweep: the parameter value, the seed and the summary."""

    parameter: str
    value: float | int | str
    seed: int
    summary: Mapping[str, Any]

    def flat(self) -> dict[str, Any]:
        row: dict[str, Any] = {"parameter": self.parameter, "value": self.value, "seed": self.seed}
        row.update({k: v for k, v in self.summary.items() if k not in ("seed", "action_counts")})
        return row


def sweep(
    model: DiscreteGenerativeModel,
    config: InferenceConfig,
    parameter: str,
    values: Sequence[float | int | str],
    *,
    steps: int,
    seeds: Sequence[int] = (0,),
    process: DiscreteGenerativeModel | None = None,
) -> list[SweepRow]:
    """Run one episode per ``(value, seed)`` and return their summaries.

    ``parameter`` is an :class:`InferenceConfig` field from :data:`SWEEPABLE`.
    Every value is validated by ``InferenceConfig`` before any episode runs, so
    a bad value in the middle of the list fails immediately.
    """
    if parameter not in SWEEPABLE:
        raise ValueError(f"Cannot sweep {parameter!r}; choose from {sorted(SWEEPABLE)}")
    if not values:
        raise ValueError("values must not be empty")
    if not seeds:
        raise ValueError("seeds must not be empty")
    kind = SWEEPABLE[parameter]
    typed = [_typed(parameter, value, kind) for value in values]
    configs = [replace(config, **{parameter: value}) for value in typed]
    rows: list[SweepRow] = []
    for value, varied in zip(typed, configs, strict=True):
        for seed in seeds:
            trajectory = run_episode(model, varied, steps, seed=seed, process=process)
            rows.append(SweepRow(parameter, value, int(seed), trajectory.summary()))
    return rows


def sweep_model(
    factory: Callable[[float], DiscreteGenerativeModel],
    parameter: str,
    values: Sequence[float],
    config: InferenceConfig,
    *,
    steps: int,
    seeds: Sequence[int] = (0,),
) -> list[SweepRow]:
    """Like :func:`sweep`, but vary the *world and agent model* instead of the config.

    ``factory(value)`` builds the model for each value (for example
    ``lambda noise: example_model(sensor_noise=noise)``); the agent has a
    correct model of that world. ``parameter`` is only a label for the rows.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not seeds:
        raise ValueError("seeds must not be empty")
    rows: list[SweepRow] = []
    for value in values:
        model = factory(float(value))
        for seed in seeds:
            trajectory = run_episode(model, config, steps, seed=seed)
            rows.append(SweepRow(parameter, float(value), int(seed), trajectory.summary()))
    return rows


def _typed(parameter: str, value: Any, kind: type) -> Any:
    if kind is str:
        return str(value)
    if kind is int:
        if isinstance(value, bool) or int(value) != float(value):
            raise ValueError(f"{parameter} needs integer values, got {value!r}")
        return int(value)
    return float(value)


def summarize_sweep(rows: Sequence[SweepRow]) -> list[dict[str, Any]]:
    """Mean and standard deviation over seeds of each numeric metric, per value."""
    grouped: dict[Any, list[SweepRow]] = {}
    for row in rows:
        grouped.setdefault(row.value, []).append(row)
    metrics = (
        "mean_free_energy",
        "mean_complexity",
        "mean_accuracy",
        "belief_accuracy",
        "preference_rate",
        "mean_policy_entropy",
    )
    table: list[dict[str, Any]] = []
    for value, members in grouped.items():
        entry: dict[str, Any] = {
            "parameter": members[0].parameter,
            "value": value,
            "episodes": len(members),
        }
        for metric in metrics:
            samples = np.array([float(member.summary[metric]) for member in members])
            entry[metric] = float(np.mean(samples))
            entry[f"{metric}_std"] = float(np.std(samples))
        table.append(entry)
    return table


def save_table(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write dictionaries with identical keys as CSV."""
    if not rows:
        raise ValueError("rows must not be empty")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination
