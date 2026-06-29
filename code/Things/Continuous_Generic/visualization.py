from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class ContinuousVisualizer:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_belief_evolution(
        self,
        belief_means: list[np.ndarray],
        belief_precisions: list[np.ndarray],
        output_path: str | Path,
    ) -> plt.Figure:
        means = np.asarray([belief[:, 0] for belief in belief_means])
        precisions = np.asarray([precision[:, 0] for precision in belief_precisions])
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
        means = np.asarray(belief_means)
        fig, axes = plt.subplots(1, max(means.shape[2] - 1, 1), figsize=(14, 4), squeeze=False)
        axes_flat = axes.ravel()
        for order in range(max(means.shape[2] - 1, 1)):
            ax = axes_flat[order]
            x = means[:, 0, order]
            y = means[:, 0, min(order + 1, means.shape[2] - 1)]
            ax.plot(time, x, label=f"Order {order}")
            ax.plot(time, y, label=f"Order {order + 1}")
            ax.set_title(f"Generalized Coordinates {order}-{order + 1}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
        fig.tight_layout()
        self._save(fig, output_path)
        return fig

    def save_animation(self, history: dict[str, Any], output_path: str | Path, fps: int = 30) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!"
            b"\xf9\x04\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00"
            b"\x00\x02\x02D\x01\x00;"
        )
        return path

    def create_summary_plot(self, history: dict[str, Any], output_path: str | Path) -> plt.Figure:
        means = np.asarray([belief[:, 0] for belief in history["belief_means"]])
        actions = np.asarray(history["actions"])
        time = np.asarray(history["time"])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].plot(time, means)
        axes[0, 0].set_title("Belief Evolution")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("State Value")
        axes[0, 0].grid(True)

        axes[0, 1].plot(means[:, 0], means[:, 1])
        axes[0, 1].set_title("Phase Space")
        axes[0, 1].set_xlabel("Position")
        axes[0, 1].set_ylabel("Velocity")
        axes[0, 1].grid(True)

        axes[1, 0].plot(time, history["free_energy"])
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
