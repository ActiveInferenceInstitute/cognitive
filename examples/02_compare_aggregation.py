"""Compare how policies sharing a first action are combined, on a corridor with a goal."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from cognitive.settings import load_settings
from cognitive.simulation import corridor_model, save_table, summarize_sweep, sweep
from cognitive.visualization.trajectory_plots import plot_sweep, save_figure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("output/examples/02_aggregation"))
parser.add_argument("--seeds", type=int, default=20)
args = parser.parse_args()

settings = load_settings(environ={})
rows = sweep(
    corridor_model(length=4, slip=0.1),
    settings.inference_config(temporal_horizon=3, temperature=0.5),
    "policy_aggregation",
    ["marginal", "min"],
    steps=settings.simulation.steps,
    seeds=range(args.seeds),
)
summary = summarize_sweep(rows)
save_table(summary, args.output / "summary.csv")
save_figure(plot_sweep(summary), args.output / "aggregation.png")
for row in summary:
    print(row["value"], round(row["preference_rate"], 3), round(row["belief_accuracy"], 3))
