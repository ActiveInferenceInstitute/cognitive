"""Figures for episodes, expected free energy, sweeps and generative models.

Every ``plot_*`` function takes recorded data and returns a Matplotlib
``Figure`` without writing anything or touching the global backend, so a
caller decides when to show, save or close it. :func:`save_figure` writes a
figure and closes it. The colour choices follow one rule across the figures:
beliefs and probabilities use a sequential map, the three expected-free-energy
parts use fixed colours (risk, ambiguity, information gain), and the agent's
own choices are drawn in the same accent wherever they appear.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..models.active_inference.generative_model import DiscreteGenerativeModel
from ..simulation import Trajectory

CHOICE = "#c2410c"
RISK, AMBIGUITY, GAIN, TOTAL = "#2563eb", "#7c3aed", "#059669", "#111827"
SEQUENTIAL = "viridis"


def _labels(given: Sequence[str], count: int, prefix: str) -> list[str]:
    return list(given) if len(given) == count else [f"{prefix}{i}" for i in range(count)]


def _at(names: Sequence[str], index: int) -> str:
    return names[index] if index < len(names) else ""


def plot_trajectory(trajectory: Trajectory) -> Figure:
    """Beliefs, choices, free energy and policy posteriors of one episode."""
    steps = trajectory.steps
    states = _labels(trajectory.state_labels, trajectory.beliefs.shape[1], "s")
    actions = _labels(trajectory.action_labels, trajectory.policy_posteriors.shape[1], "a")
    observations = _labels(
        trajectory.observation_labels, int(trajectory.observations.max(initial=0)) + 1, "o"
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    belief_ax, choice_ax, energy_ax, policy_ax = axes.ravel()
    image = belief_ax.imshow(
        trajectory.beliefs.T,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
        cmap=SEQUENTIAL,
        extent=(-0.5, steps + 0.5, -0.5, len(states) - 0.5),
    )
    belief_ax.plot(
        np.arange(steps + 1),
        trajectory.states,
        "o",
        color="white",
        markersize=4,
        label="true state",
    )
    belief_ax.set(title="Beliefs q(s) over time", xlabel="step", yticks=range(len(states)))
    belief_ax.set_yticklabels(states)
    belief_ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    fig.colorbar(image, ax=belief_ax, label="probability")

    time = np.arange(1, steps + 1)
    choice_ax.scatter(
        time, trajectory.actions, marker="s", color=CHOICE, label="action chosen", zorder=3
    )
    choice_ax.step(time, trajectory.observations, where="mid", color="#6b7280", alpha=0.6)
    choice_ax.scatter(time, trajectory.observations, color="#6b7280", label="observation", zorder=2)
    choice_ax.axhline(
        trajectory.preferred_observation, color=GAIN, linestyle=":", label="preferred observation"
    )
    levels = max(len(actions), len(observations))
    choice_ax.set(title="Choices and observations", xlabel="step", yticks=range(levels))
    choice_ax.set_yticklabels(
        [f"{_at(actions, level)} | {_at(observations, level)}" for level in range(levels)]
    )
    choice_ax.legend(loc="best", fontsize=8)

    energy_ax.plot(time, trajectory.free_energy, color=TOTAL, linewidth=2, label="free energy F")
    energy_ax.plot(time, trajectory.complexity, color=AMBIGUITY, label="complexity")
    energy_ax.plot(time, -trajectory.accuracy, color=RISK, label="-accuracy")
    energy_ax.set(title="Variational free energy per step", xlabel="step", ylabel="nats")
    energy_ax.legend(loc="best", fontsize=8)

    image = policy_ax.imshow(
        trajectory.policy_posteriors.T,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
        cmap=SEQUENTIAL,
        extent=(0.5, steps + 0.5, -0.5, len(actions) - 0.5),
    )
    policy_ax.plot(time, trajectory.actions, "s", color=CHOICE, markersize=4, label="chosen")
    policy_ax.set(title="Action posterior P(a)", xlabel="step", yticks=range(len(actions)))
    policy_ax.set_yticklabels(actions)
    policy_ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    fig.colorbar(image, ax=policy_ax, label="probability")
    fig.suptitle(
        f"Episode of {steps} steps (seed {trajectory.seed}, "
        f"{trajectory.config.get('method', '?')}, "
        f"{trajectory.config.get('policy_aggregation', '?')})"
    )
    return fig


def plot_efe_components(trajectory: Trajectory, step: int = 0) -> Figure:
    """Risk, ambiguity, total and information gain of each action at one step."""
    if not 0 <= step < trajectory.steps:
        raise ValueError(f"step must be in [0, {trajectory.steps - 1}], got {step}")
    count = trajectory.efe.shape[1]
    labels = _labels(trajectory.action_labels, count, "a")
    positions = np.arange(count)
    width = 0.2
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for offset, (name, values, colour) in enumerate(
        (
            ("risk", trajectory.risk[step], RISK),
            ("ambiguity", trajectory.ambiguity[step], AMBIGUITY),
            ("expected free energy G", trajectory.efe[step], TOTAL),
            ("information gain (included in ambiguity)", trajectory.epistemic[step], GAIN),
        )
    ):
        ax.bar(positions + (offset - 1.5) * width, values, width, label=name, color=colour)
    chosen = int(trajectory.actions[step])
    ax.set(
        title=f"Expected free energy by action at step {step}",
        ylabel="nats",
        xticks=positions,
    )
    ax.set_xticklabels(
        [f"{name} (chosen)" if i == chosen else name for i, name in enumerate(labels)]
    )
    ax.get_xticklabels()[chosen].set_color(CHOICE)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    return fig


def plot_sweep(
    summary: Sequence[Mapping[str, Any]],
    metrics: Sequence[str] = ("preference_rate", "belief_accuracy", "mean_free_energy"),
) -> Figure:
    """One panel per metric against the swept value, with a band for the seed spread."""
    if not summary:
        raise ValueError("summary must not be empty")
    parameter = str(summary[0]["parameter"])
    values = [row["value"] for row in summary]
    numeric = all(isinstance(value, (int, float)) for value in values)
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(4.2 * len(metrics), 3.8), constrained_layout=True, squeeze=False
    )
    for ax, metric in zip(axes[0], metrics, strict=True):
        means = np.array([float(row[metric]) for row in summary])
        spread = np.array([float(row.get(f"{metric}_std", 0.0)) for row in summary])
        if numeric:
            order = np.argsort(values)
            x = np.asarray(values, dtype=float)[order]
            ax.plot(x, means[order], marker="o", color=TOTAL)
            ax.fill_between(
                x, (means - spread)[order], (means + spread)[order], color=RISK, alpha=0.2
            )
        else:
            ax.bar(range(len(values)), means, yerr=spread, color=RISK, alpha=0.8, capsize=3)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([str(value) for value in values])
        ax.set(title=metric.replace("_", " "), xlabel=parameter)
    episodes = summary[0].get("episodes")
    fig.suptitle(
        f"Sweep of {parameter}" + (f" ({episodes} episodes per value)" if episodes else "")
    )
    return fig


def plot_model(model: DiscreteGenerativeModel) -> Figure:
    """The likelihood A, one transition matrix per action, preferences C and prior D."""
    actions = model.num_actions
    fig, axes = plt.subplots(
        1, actions + 3, figsize=(3.0 * (actions + 3), 3.4), constrained_layout=True
    )

    def draw(ax: Any, matrix: np.ndarray, title: str) -> None:
        shown = np.atleast_2d(matrix)
        top = max(float(shown.max()), 1e-9)
        ax.imshow(shown, vmin=0, vmax=top, cmap=SEQUENTIAL)
        ax.set_title(title)
        for (row, col), value in np.ndenumerate(shown):
            # Dark text on the light end of the colour map, light text on the dark end.
            ink = "black" if value > 0.6 * top else "white"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=ink, fontsize=8)
        ax.set_xticks(range(shown.shape[1]))
        ax.set_yticks(range(shown.shape[0]))

    draw(axes[0], model.A, "A  P(o | s)")
    for action in range(actions):
        draw(axes[1 + action], model.B[:, :, action], f"B  P(s' | s), a={action}")
    draw(axes[-2], model.preferences[:, None], "C  preferences")
    draw(axes[-1], model.D[:, None], "D  prior over s")
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    """Write ``fig`` (format from the suffix), create the folder, and close it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(destination, dpi=dpi)
    finally:
        plt.close(fig)
    return destination
