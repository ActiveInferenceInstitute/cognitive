from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from cognitive.models.active_inference.generative_model import DiscreteGenerativeModel

EPS = 1e-12


@dataclass
class SimplePOMDPState:
    current_state: int
    beliefs: np.ndarray
    time_step: int = 0
    history: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "states": [],
            "observations": [],
            "actions": [],
            "beliefs": [],
            "free_energy": [],
            "efe_total": [],
            "efe_epistemic": [],
            "efe_pragmatic": [],
        }
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Probability vector must be finite and non-empty")
    vector = np.maximum(vector, 0.0)
    total = float(vector.sum())
    if total <= EPS:
        return np.ones_like(vector, dtype=float) / vector.size
    return vector / total


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temperature = float(temperature)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    scaled = np.asarray(values, dtype=float) / temperature
    shifted = scaled - np.max(scaled)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def compute_expected_free_energy(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    beliefs: np.ndarray,
    action: int,
) -> tuple[float, float, float]:
    model = DiscreteGenerativeModel(
        np.asarray(A, dtype=float),
        np.asarray(B, dtype=float),
        np.asarray(C, dtype=float),
        np.full(np.asarray(B).shape[0], 1.0 / np.asarray(B).shape[0]),
        np.full(np.asarray(B).shape[2], 1.0 / np.asarray(B).shape[2]),
    )
    total, risk, ambiguity, epistemic_gain = model.expected_free_energy(beliefs, action)
    pragmatic = risk + ambiguity
    return float(total), float(epistemic_gain), float(pragmatic)


class SimplePOMDP:
    REQUIRED_SECTIONS = {
        "model",
        "state_space",
        "observation_space",
        "action_space",
        "matrices",
        "inference",
        "visualization",
    }

    def __init__(self, config: str | Path | dict[str, Any]):
        self.config = self._load_config(config)
        self._validate_config()
        self.rng = np.random.default_rng(self.config.get("seed", 7))

        self.A = self._initialize_matrix("A_matrix")
        self.B = self._initialize_matrix("B_matrix")
        self.C = self._initialize_matrix("C_matrix")
        self.D = self._initialize_matrix("D_matrix")
        self.E = self._initialize_matrix("E_matrix")

        self._validate_matrices()

        initial_state = int(self.config["state_space"].get("initial_state", 0))
        self.state = SimplePOMDPState(current_state=initial_state, beliefs=self.D.copy())
        self.plotter = _SimplePOMDPPlotter(self)

    def _load_config(self, config: str | Path | dict[str, Any]) -> dict[str, Any]:
        if isinstance(config, dict):
            return dict(config)
        config_path = Path(config).expanduser().resolve()
        with config_path.open(encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
        if not isinstance(loaded, dict):
            raise ValueError("Configuration file must contain a mapping")
        visualization = loaded.get("visualization")
        if isinstance(visualization, dict) and isinstance(visualization.get("output_dir"), str):
            output_path = Path(visualization["output_dir"]).expanduser()
            if not output_path.is_absolute():
                visualization["output_dir"] = str((config_path.parent / output_path).resolve())
        return loaded

    def _validate_config(self) -> None:
        missing = self.REQUIRED_SECTIONS.difference(self.config)
        if missing:
            raise ValueError(f"Missing required configuration sections: {sorted(missing)}")
        allowed_root = self.REQUIRED_SECTIONS | {"seed"}
        unknown_root = set(self.config).difference(allowed_root)
        if unknown_root:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown_root)}")
        self._validate_section("model", {"name", "description", "version"})
        self._validate_section("state_space", {"num_states", "state_labels", "initial_state"})
        self._validate_section("observation_space", {"num_observations", "observation_labels"})
        self._validate_section("action_space", {"num_actions", "action_labels"})
        self._validate_section(
            "inference",
            {
                "time_horizon",
                "temporal_horizon",
                "num_iterations",
                "learning_rate",
                "temperature",
                "policy_learning_rate",
                "seed",
            },
        )
        self._validate_section("visualization", {"output_dir", "formats", "dpi", "style"})
        matrices = self.config.get("matrices")
        if not isinstance(matrices, dict):
            raise ValueError("matrices must be a mapping")
        expected_matrices = {"A_matrix", "B_matrix", "C_matrix", "D_matrix", "E_matrix"}
        if set(matrices) != expected_matrices:
            raise ValueError(
                "matrices must define exactly A_matrix, B_matrix, C_matrix, D_matrix, and E_matrix"
            )
        for name, spec in matrices.items():
            if not isinstance(spec, dict):
                raise ValueError(f"{name} must be a mapping")
            allowed = {
                "shape",
                "initialization",
                "initialization_params",
                "constraints",
                "description",
                "learning_rate",
            }
            unknown = set(spec).difference(allowed)
            if unknown:
                raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")

    def _validate_section(self, name: str, allowed: set[str]) -> None:
        value = self.config.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")

    def _initialize_matrix(self, name: str) -> np.ndarray:
        spec = self.config["matrices"][name]
        shape = tuple(spec["shape"])
        if not shape or any(int(value) < 1 for value in shape):
            raise ValueError(f"{name} must have a positive shape")
        init = spec.get("initialization", "uniform")
        params = spec.get("initialization_params", {})

        if init in {"random", "random_stochastic"}:
            matrix = self.rng.random(shape)
        elif init == "identity_based":
            matrix = self._identity_based(shape, float(params.get("strength", 0.8)))
        elif init == "log_preferences":
            preferences = np.asarray(params.get("preferences", np.zeros(shape)), dtype=float)
            if preferences.shape != shape:
                raise ValueError(f"{name} preferences must have shape {shape}")
            matrix = preferences
        elif init == "uniform":
            matrix = np.ones(shape, dtype=float) / float(np.prod(shape))
        else:
            raise ValueError(f"Unsupported initialization for {name}: {init}")

        for constraint in spec.get("constraints", []):
            matrix = self._apply_constraint(matrix, constraint)
        return np.asarray(matrix, dtype=float)

    def _identity_based(self, shape: tuple[int, ...], strength: float) -> np.ndarray:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("identity strength must be in [0, 1]")
        if len(shape) == 2:
            n_rows, n_cols = shape
            matrix = np.zeros(shape, dtype=float)
            if n_rows == 1:
                return np.ones(shape, dtype=float)
            for col in range(n_cols):
                matrix[col % n_rows, col] = strength
                remainder = max(1.0 - strength, 0.0)
                if n_rows > 1:
                    matrix[:, col] += remainder / (n_rows - 1)
                    matrix[col % n_rows, col] = strength
            return self._column_normalize(matrix)

        if len(shape) == 3:
            states, previous_states, actions = shape
            tensor = np.zeros(shape, dtype=float)
            for action in range(actions):
                for previous_state in range(previous_states):
                    target = (previous_state + action) % states
                    tensor[target, previous_state, action] = strength
                    remainder = max(1.0 - strength, 0.0)
                    if states > 1:
                        tensor[:, previous_state, action] += remainder / (states - 1)
                        tensor[target, previous_state, action] = strength
            return self._normalize_transition_tensor(tensor)

        raise ValueError("identity_based initialization supports 2D or 3D shapes")

    def _apply_constraint(self, matrix: np.ndarray, constraint: str) -> np.ndarray:
        if constraint == "non_negative":
            return np.maximum(matrix, 0.0)
        if constraint == "column_stochastic":
            if matrix.ndim == 3:
                return self._normalize_transition_tensor(matrix)
            return self._column_normalize(matrix)
        if constraint == "row_stochastic":
            if matrix.ndim == 3:
                raise ValueError("B_matrix uses column_stochastic, not row_stochastic")
            return self._row_normalize(matrix)
        if constraint in {"sum_to_one", "non_negative"}:
            if constraint == "sum_to_one":
                total = float(matrix.sum())
                if total <= EPS:
                    raise ValueError("sum_to_one cannot normalize an all-zero matrix")
                return matrix / total
            return np.maximum(matrix, 0.0)
        raise ValueError(f"Unsupported matrix constraint: {constraint}")

    @staticmethod
    def _column_normalize(matrix: np.ndarray) -> np.ndarray:
        matrix = np.maximum(matrix, 0.0)
        sums = matrix.sum(axis=0, keepdims=True)
        sums = np.where(sums <= EPS, 1.0, sums)
        return matrix / sums

    @staticmethod
    def _row_normalize(matrix: np.ndarray) -> np.ndarray:
        matrix = np.maximum(matrix, 0.0)
        sums = matrix.sum(axis=1, keepdims=True)
        sums = np.where(sums <= EPS, 1.0, sums)
        return matrix / sums

    @classmethod
    def _normalize_transition_tensor(cls, tensor: np.ndarray) -> np.ndarray:
        tensor = np.maximum(tensor, 0.0)
        normalized = np.empty_like(tensor, dtype=float)
        for action in range(tensor.shape[2]):
            normalized[:, :, action] = cls._column_normalize(tensor[:, :, action])
        return normalized

    def _validate_matrices(self) -> None:
        num_observations = self.config["observation_space"]["num_observations"]
        num_states = self.config["state_space"]["num_states"]
        num_actions = self.config["action_space"]["num_actions"]

        if self.A.shape != (num_observations, num_states):
            raise ValueError("A matrix shape must match observation and state spaces")
        if (
            not np.all(np.isfinite(self.A))
            or not np.all(self.A >= 0)
            or not np.allclose(self.A.sum(axis=0), 1.0)
        ):
            raise ValueError("A matrix must be column stochastic")

        if self.B.shape != (num_states, num_states, num_actions):
            raise ValueError("B matrix shape must match state and action spaces")
        for action in range(num_actions):
            if (
                not np.all(np.isfinite(self.B[:, :, action]))
                or not np.all(self.B[:, :, action] >= 0)
                or not np.allclose(self.B[:, :, action].sum(axis=0), 1.0)
            ):
                raise ValueError("B matrix must be column stochastic")

        if self.C.shape != (num_observations,):
            raise ValueError("C matrix must define one preference per observation")
        if not np.all(np.isfinite(self.C)):
            raise ValueError("C matrix must contain finite preferences")
        if self.D.shape != (num_states,):
            raise ValueError("D matrix must define one prior per state")
        self.D = _normalize(self.D)
        if self.E.shape != (num_actions,):
            raise ValueError("E matrix must define one prior per action")
        self.E = _normalize(self.E)
        initial_state = int(self.config["state_space"].get("initial_state", 0))
        if not 0 <= initial_state < num_states:
            raise ValueError("initial_state is out of range")

    def step(self, action: int | None = None) -> tuple[int, float]:
        if action is None:
            action, _ = self._select_action()
        else:
            action = int(action)
        if not 0 <= action < self.config["action_space"]["num_actions"]:
            raise ValueError(f"Action out of range: {action}")

        transition_probs = self.B[:, self.state.current_state, action]
        next_state = int(self.rng.choice(len(transition_probs), p=_normalize(transition_probs)))
        observation = self._get_observation(next_state)
        free_energy = self._update_beliefs(observation, action)

        total, epistemic, pragmatic = self._expected_free_energy_components()
        self.state.current_state = next_state
        self.state.time_step += 1
        self.state.history["states"].append(next_state)
        self.state.history["observations"].append(observation)
        self.state.history["actions"].append(action)
        self.state.history["beliefs"].append(self.state.beliefs.copy())
        self.state.history["free_energy"].append(float(free_energy))
        self.state.history["efe_total"].append(total.copy())
        self.state.history["efe_epistemic"].append(epistemic.copy())
        self.state.history["efe_pragmatic"].append(pragmatic.copy())

        return observation, float(free_energy)

    def run(self, steps: int, actions: list[int] | None = None) -> list[tuple[int, float]]:
        """Run a deterministic-length simulation and return observations/free energies."""
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if actions is not None and len(actions) != steps:
            raise ValueError("actions must contain exactly one action per step")
        return [self.step(None if actions is None else actions[index]) for index in range(steps)]

    def reset(self) -> None:
        """Reset beliefs, time, and history to the configured prior."""
        initial_state = int(self.config["state_space"].get("initial_state", 0))
        self.state = SimplePOMDPState(initial_state, self.D.copy())

    def save_state(self, path: str | Path) -> Path:
        """Persist model state and history as versioned YAML."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "current_state": self.state.current_state,
            "beliefs": self.state.beliefs.tolist(),
            "time_step": self.state.time_step,
            "history": {
                key: [
                    value.tolist() if isinstance(value, np.ndarray) else value for value in values
                ]
                for key, values in self.state.history.items()
            },
        }
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("w", encoding="utf-8") as state_file:
            yaml.safe_dump(payload, state_file, sort_keys=True)
        temporary.replace(destination)
        return destination

    def load_state(self, path: str | Path) -> None:
        """Load a validated model state and history."""
        with Path(path).open(encoding="utf-8") as state_file:
            payload = yaml.safe_load(state_file)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported or malformed POMDP state")
        current_state = int(payload["current_state"])
        beliefs = _normalize(np.asarray(payload["beliefs"], dtype=float))
        if current_state < 0 or current_state >= self.B.shape[0] or beliefs.shape != self.D.shape:
            raise ValueError("Saved state dimensions are incompatible with this model")
        history = payload.get("history")
        if not isinstance(history, dict) or set(history) != set(self.state.history):
            raise ValueError("Saved history fields are incompatible with this model")
        self.state = SimplePOMDPState(
            current_state=current_state,
            beliefs=beliefs,
            time_step=int(payload["time_step"]),
            history={key: list(values) for key, values in history.items()},
        )

    def _get_observation(self, state: int) -> int:
        probabilities = _normalize(self.A[:, state])
        return int(self.rng.choice(len(probabilities), p=probabilities))

    def _update_beliefs(self, observation: int, action: int) -> float:
        predicted = self.B[:, :, action] @ self.state.beliefs
        likelihood = self.A[observation, :]
        posterior = _normalize(predicted * likelihood)
        learning_rate = float(self.config["inference"].get("learning_rate", 1.0))
        learning_rate = float(np.clip(learning_rate, 0.0, 1.0))
        self.state.beliefs = _normalize(
            (1.0 - learning_rate) * self.state.beliefs + learning_rate * posterior
        )
        return -float(np.log(np.dot(likelihood, predicted) + EPS))

    def _select_action(self) -> tuple[int, np.ndarray]:
        total, _, _ = self._expected_free_energy_components()
        action_prior = _normalize(self.E)
        scores = total - np.log(action_prior + EPS)
        return int(np.argmin(scores)), total

    def _expected_free_energy_components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        totals: list[float] = []
        epistemic_values: list[float] = []
        pragmatic_values: list[float] = []
        for action in range(self.config["action_space"]["num_actions"]):
            total, epistemic, pragmatic = compute_expected_free_energy(
                self.A, self.B, self.C, self.state.beliefs, action
            )
            totals.append(total)
            epistemic_values.append(epistemic)
            pragmatic_values.append(pragmatic)
        return np.array(totals), np.array(epistemic_values), np.array(pragmatic_values)

    def visualize(self, plot_type: str, **kwargs: Any) -> plt.Figure:
        return self.plotter.visualize(plot_type, **kwargs)


class _SimplePOMDPPlotter:
    def __init__(self, model: SimplePOMDP):
        self.model = model
        self.output_dir = Path(model.config["visualization"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize(self, plot_type: str, **_: Any) -> plt.Figure:
        handlers = {
            "belief_evolution": self._plot_belief_evolution,
            "free_energy_landscape": self._plot_free_energy_landscape,
            "policy_evaluation": self._plot_policy_evaluation,
            "state_transitions": self._plot_state_transitions,
            "observation_likelihood": self._plot_observation_likelihood,
            "efe_components_detailed": self._plot_efe_components_detailed,
        }
        if plot_type not in handlers:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        fig = handlers[plot_type]()
        fig.savefig(self.output_dir / f"{plot_type}.png", bbox_inches="tight")
        plt.close(fig)
        return fig

    def _belief_history(self) -> np.ndarray:
        if self.model.state.history["beliefs"]:
            return np.asarray(self.model.state.history["beliefs"])
        return self.model.state.beliefs.reshape(1, -1)

    def _plot_belief_evolution(self) -> plt.Figure:
        beliefs = self._belief_history()
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = self.model.config["state_space"].get("state_labels", [])
        for state_idx in range(beliefs.shape[1]):
            label = labels[state_idx] if state_idx < len(labels) else f"State {state_idx}"
            ax.plot(np.arange(beliefs.shape[0]), beliefs[:, state_idx], label=label)
        ax.set_title("Belief Evolution")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Belief Probability")
        ax.legend()
        return fig

    def _plot_free_energy_landscape(self) -> plt.Figure:
        fig = plt.figure(figsize=(18, 5))
        titles = ["Total Expected Free Energy", "Epistemic Value", "Pragmatic Value"]
        values = self._belief_grid_values()
        for idx, title in enumerate(titles):
            ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
            surface = ax.plot_surface(values["x"], values["y"], values[title], cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("Belief in State 0")
            ax.set_ylabel("Belief in State 1")
            ax.set_zlabel("Value")
            fig.colorbar(surface, ax=ax, shrink=0.65)
        fig.tight_layout()
        return fig

    def _belief_grid_values(self) -> dict[str, np.ndarray]:
        x = np.linspace(0.0, 1.0, 24)
        y = np.linspace(0.0, 1.0, 24)
        X, Y = np.meshgrid(x, y)
        values = {
            "x": X,
            "y": Y,
            "Total Expected Free Energy": np.full_like(X, np.nan),
            "Epistemic Value": np.full_like(X, np.nan),
            "Pragmatic Value": np.full_like(X, np.nan),
        }
        original_beliefs = self.model.state.beliefs.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                b0 = X[i, j]
                b1 = Y[i, j]
                remainder = 1.0 - b0 - b1
                if remainder < -EPS:
                    continue
                beliefs = np.array([b0, b1, max(remainder, 0.0)])
                if beliefs.size != original_beliefs.size:
                    beliefs = np.resize(beliefs, original_beliefs.size)
                    beliefs = _normalize(beliefs)
                self.model.state.beliefs = _normalize(beliefs)
                total, epistemic, pragmatic = self.model._expected_free_energy_components()
                values["Total Expected Free Energy"][i, j] = float(np.min(total))
                values["Epistemic Value"][i, j] = float(np.mean(epistemic))
                values["Pragmatic Value"][i, j] = float(np.mean(pragmatic))
        self.model.state.beliefs = original_beliefs
        return values

    def _plot_policy_evaluation(self) -> plt.Figure:
        total, epistemic, pragmatic = self.model._expected_free_energy_components()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        series = [
            ("Total Expected Free Energy", total),
            ("Epistemic Value", epistemic),
            ("Pragmatic Value", pragmatic),
        ]
        for ax, (title, values) in zip(axes, series, strict=False):
            ax.bar(np.arange(values.size), values)
            ax.set_title(title)
            ax.set_xlabel("Policy Index")
            ax.set_ylabel("Value")
        fig.tight_layout()
        return fig

    def _plot_state_transitions(self) -> plt.Figure:
        n_actions = self.model.B.shape[2]
        fig, axes = plt.subplots(1, n_actions, figsize=(5 * n_actions, 4), squeeze=False)
        for action in range(n_actions):
            ax = axes[0, action]
            im = ax.imshow(self.model.B[:, :, action], cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"State Transitions - Action {action}")
            ax.set_xlabel("Previous State")
            ax.set_ylabel("Next State")
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return fig

    def _plot_observation_likelihood(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(self.model.A, cmap="viridis", vmin=0, vmax=1)
        ax.set_title("Observation Likelihood")
        ax.set_xlabel("State")
        ax.set_ylabel("Observation")
        fig.colorbar(im, ax=ax)
        return fig

    def _plot_efe_components_detailed(self) -> plt.Figure:
        total = np.asarray(
            self.model.state.history["efe_total"]
            or [self.model._expected_free_energy_components()[0]]
        )
        epistemic = np.asarray(
            self.model.state.history["efe_epistemic"]
            or [self.model._expected_free_energy_components()[1]]
        )
        pragmatic = np.asarray(
            self.model.state.history["efe_pragmatic"]
            or [self.model._expected_free_energy_components()[2]]
        )
        selected_total = np.min(total, axis=1)
        selected_epistemic = np.mean(epistemic, axis=1)
        selected_pragmatic = np.mean(pragmatic, axis=1)
        time = np.arange(selected_total.size)

        fig = plt.figure(figsize=(16, 12))
        axes = [fig.add_subplot(3, 2, idx + 1) for idx in range(5)]

        axes[0].plot(time, selected_total)
        axes[0].set_title("Total Expected Free Energy")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Value")

        axes[1].stackplot(
            time, selected_epistemic, selected_pragmatic, labels=["Epistemic", "Pragmatic"]
        )
        axes[1].set_title("EFE Components")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Value")
        axes[1].legend()

        ratio = selected_epistemic / (selected_pragmatic + EPS)
        axes[2].plot(time, ratio)
        axes[2].set_title("Component Ratio")
        axes[2].set_xlabel("Time Step")
        axes[2].set_ylabel("Ratio")

        scatter = axes[3].scatter(selected_epistemic, selected_pragmatic, c=time, cmap="viridis")
        axes[3].set_title("Epistemic vs Pragmatic Value")
        axes[3].set_xlabel("Epistemic Value")
        axes[3].set_ylabel("Pragmatic Value")

        window = min(5, selected_total.size)
        kernel = np.ones(window) / window
        running = np.convolve(selected_total, kernel, mode="same")
        axes[4].plot(time, running)
        axes[4].set_title("Running Averages")
        axes[4].set_xlabel("Time Step")
        axes[4].set_ylabel("Value")

        fig.colorbar(scatter, ax=axes[3])
        fig.tight_layout()
        return fig
