"""An agent whose model of the world is wrong: believed dynamics versus real dynamics."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from cognitive.settings import load_settings
from cognitive.simulation import example_model, run_episode
from cognitive.visualization.trajectory_plots import plot_trajectory, save_figure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("output/examples/04_mismatch"))
args = parser.parse_args()

settings = load_settings(environ={})
believed = example_model(sensor_noise=0.1, stickiness=0.9)
real = example_model(sensor_noise=0.1, stickiness=0.55)  # the world is far less predictable
config = settings.inference_config()
for name, world in (("correct_model", believed), ("wrong_model", real)):
    trajectory = run_episode(believed, config, settings.simulation.steps, seed=1, process=world)
    save_figure(plot_trajectory(trajectory), args.output / f"{name}.png")
    print(name, round(trajectory.belief_accuracy, 3), round(trajectory.preference_rate, 3))
