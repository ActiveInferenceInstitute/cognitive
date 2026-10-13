"""Run two controllers at different timescales on a corridor with a preferred endpoint."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from cognitive.episodes import corridor_hierarchy, run_hierarchical_episode
from cognitive.settings import load_settings
from cognitive.visualization.trajectory_plots import plot_hierarchical_trajectory, save_figure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("output/examples/08_hierarchy"))
args = parser.parse_args()
settings = load_settings(environ={})
trajectory = run_hierarchical_episode(
    corridor_hierarchy(), settings.inference_config(), settings.inference_config(), 30, seed=7
)
trajectory.save_json(args.output / "hierarchy.json")
trajectory.lower.save_json(args.output / "trajectory.json")
save_figure(plot_hierarchical_trajectory(trajectory), args.output / "episode.png")
