"""Plots and animations for continuous active inference histories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


class ContinuousVisualizer:
    """Render validated continuous-agent histories to image files."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _history_means(history: dict[str, Any]) -> np.ndarray:
        if "belief_means" not in history:
            raise ValueError("history must contain belief_means")
        means = np.asarray(history["belief_means"], dtype=float)
        if means.ndim != 3 or means.shape[0] == 0 or means.shape[2] == 0:
            raise ValueError("belief_means must have shape (frames, states, orders)")
        if not np.all(np.isfinite(means)):
            raise ValueError("belief_means must be finite")
        return means

    @staticmethod
    def _history_time(history: dict[str, Any], frames: int) -> np.ndarray:
        time = np.asarray(history.get("time", np.arange(frames)), dtype=float)
        if time.shape != (frames,) or not np.all(np.isfinite(time)):
            raise ValueError("time must contain one finite value per frame")
        return time

    def plot_belief_evolution(
        self,
        belief_means: list[np.ndarray],
        belief_precisions: list[np.ndarray],
        output_path: str | Path,
    ) -> plt.Figure:
        means_full = np.asarray(belief_means, dtype=float)
        precisions_full = np.asarray(belief_precisions, dtype=float)
        if means_full.ndim != 3 or precisions_full.shape != means_full.shape:
            raise ValueError("belief means and precisions must share a 3-D shape")
        means = means_full[:, :, 0]
        precisions = precisions_full[:, :, 0]
        time = np.arange(means.shape[0])

        fig, ax = plt.subplots(figsize=(12, 6))
        for idx in range(means.shape[1]):
            std = 1.0 / np.sqrt(np.maximum(precisions[:, idx], 1e-12))
            ax.plot(time, means[:, idx], label=f"State {idx + 1}")
            ax.fill_between(time, means[:, idx] - std, means[:, idx] + std, alpha=0.2)
        ax.set_title("Belief Evolution")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Belief Mean")
        ax.legend()
        ax.grid(True)
        self._save(fig, output_path)
        return fig

    def plot_generalized_coordinates_relationships(
        self,
        belief_means: list[np.ndarray],
        time: np.ndarray,
        output_path: str | Path,
    ) -> plt.Figure:
        means = self._history_means({"belief_means": belief_means})
        time_values = self._history_time({"time": time}, means.shape[0])
        fig, axes = plt.subplots(1, max(means.shape[2] - 1, 1), figsize=(14, 4), squeeze=False)
        axes_flat = axes.ravel()
        for order in range(max(means.shape[2] - 1, 1)):
            ax = axes_flat[order]
            x = means[:, 0, order]
            y = means[:, 0, min(order + 1, means.shape[2] - 1)]
            ax.plot(time_values, x, label=f"Order {order}")
            ax.plot(time_values, y, label=f"Order {order + 1}")
            ax.set_title(f"Generalized Coordinates {order}-{order + 1}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
        fig.tight_layout()
        self._save(fig, output_path)
        return fig

    def save_animation(
        self, history: dict[str, Any], output_path: str | Path, fps: int = 30
    ) -> Path:
        """Save one rendered frame per history sample as a Pillow GIF."""
        if not isinstance(fps, int) or fps < 1:
            raise ValueError("fps must be a positive integer")
        means = self._history_means(history)
        time = self._history_time(history, means.shape[0])
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        lines = [ax.plot([], [], label=f"State {idx + 1}")[0] for idx in range(means.shape[1])]
        ax.set_xlim(
            float(time.min()),
            float(time.max()) if time.max() > time.min() else float(time.min() + 1.0),
        )
        lower = float(means[:, :, 0].min())
        upper = float(means[:, :, 0].max())
        margin = max((upper - lower) * 0.1, 1e-3)
        ax.set_ylim(lower - margin, upper + margin)
        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        ax.set_title("Continuous Belief Evolution")
        ax.grid(True)
        ax.legend()

        def update(frame: int) -> list[Any]:
            for state, line in enumerate(lines):
                line.set_data(time[: frame + 1], means[: frame + 1, state, 0])
            return lines

        animation = FuncAnimation(
            fig, update, frames=means.shape[0], interval=1000 / fps, blit=True, repeat=False
        )
        animation.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return path

    def create_summary_plot(self, history: dict[str, Any], output_path: str | Path) -> plt.Figure:
        means_full = self._history_means(history)
        means = means_full[:, :, 0]
        actions = np.asarray(history.get("actions"), dtype=float)
        time = self._history_time(history, means.shape[0])
        free_energy = np.asarray(history.get("free_energy"), dtype=float)
        if actions.shape[0] != means.shape[0] or free_energy.shape != (means.shape[0],):
            raise ValueError("actions and free_energy must match the history length")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].plot(time, means)
        axes[0, 0].set_title("Belief Evolution")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("State Value")
        axes[0, 0].grid(True)
        if means.shape[1] >= 2:
            axes[0, 1].plot(means[:, 0], means[:, 1])
        axes[0, 1].set_title("Phase Space")
        axes[0, 1].set_xlabel("Position")
        axes[0, 1].set_ylabel("Velocity")
        axes[0, 1].grid(True)
        axes[1, 0].plot(time, free_energy)
        axes[1, 0].set_title("Free Energy")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Value")
        axes[1, 0].grid(True)
        axes[1, 1].plot(time, actions)
        axes[1, 1].set_title("Action Evolution")
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Action")
        axes[1, 1].grid(True)
        fig.tight_layout()
        self._save(fig, output_path)
        return fig

    def _save(self, fig: plt.Figure, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
