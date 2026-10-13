"""How a noisier sensor changes what the agent believes and how often it gets what it prefers."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from cognitive.settings import load_settings
from cognitive.simulation import example_model, save_table, summarize_sweep, sweep_model
from cognitive.visualization.trajectory_plots import plot_sweep, save_figure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("output/examples/03_sensor_noise"))
parser.add_argument("--seeds", type=int, default=10)
args = parser.parse_args()

settings = load_settings(environ={})
rows = sweep_model(
    lambda noise: example_model(sensor_noise=noise),
    "sensor_noise",
    [0.05, 0.1, 0.2, 0.3, 0.4],
    settings.inference_config(),
    steps=settings.simulation.steps,
    seeds=range(args.seeds),
)
summary = summarize_sweep(rows)
save_table(summary, args.output / "summary.csv")
save_figure(plot_sweep(summary), args.output / "sensor_noise.png")
print(*[(r["value"], round(r["belief_accuracy"], 3)) for r in summary])
