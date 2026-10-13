"""Run one episode of the two-state model and draw it."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from cognitive.settings import load_settings
from cognitive.simulation import load_model, run_episode
from cognitive.visualization.trajectory_plots import (
    plot_efe_components,
    plot_trajectory,
    save_figure,
)

HERE = Path(__file__).parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("output/examples/01_episode"))
args = parser.parse_args()

settings = load_settings(HERE / "configs" / "settings.yaml", environ={})
model, labels = load_model(HERE / "configs" / "two_state.yaml")
trajectory = run_episode(
    model, settings.inference_config(), settings.simulation.steps, labels=labels
)
trajectory.save_json(args.output / "trajectory.json")
save_figure(plot_trajectory(trajectory), args.output / "episode.png", settings.output.dpi)
save_figure(plot_efe_components(trajectory, 0), args.output / "efe_step0.png", settings.output.dpi)
print(trajectory.summary())
