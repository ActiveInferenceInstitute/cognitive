"""Run a continuous agent against a controlled world and save its recorded episode."""

import argparse
from pathlib import Path

import matplotlib
from Things.Continuous_Generic import ContinuousActiveInference

matplotlib.use("Agg")
from cognitive.episodes import run_continuous_episode
from cognitive.visualization.trajectory_plots import plot_continuous_trajectory, save_figure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("output/examples/07_continuous"))
args = parser.parse_args()
trajectory = run_continuous_episode(ContinuousActiveInference(), 100, seed=7, sensor_noise=0.05)
trajectory.save_json(args.output / "trajectory.json")
trajectory.save_csv(args.output / "trajectory.csv")
save_figure(plot_continuous_trajectory(trajectory), args.output / "episode.png")
