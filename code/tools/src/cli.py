"""The ``cognitive`` command line: settings, simulation, sweeps and benchmarks.

Each subcommand is a thin wrapper that resolves :mod:`cognitive.settings`,
calls one function of :mod:`cognitive.simulation`, and writes files:

``cognitive settings``
    Print the settings that would be used, after the YAML file, environment
    variables and ``--set`` overrides are applied.
``cognitive simulate``
    Run one episode; write ``trajectory.json``, ``trajectory.csv`` and figures.
``cognitive sweep``
    Vary one inference parameter over several values and seeds; write the
    per-episode rows, the per-value summary and a figure.
``cognitive benchmark``
    Time the runtime components (see :mod:`cognitive.benchmarks`).

Exit status is ``0`` on success and ``2`` when the input is rejected (bad
settings, unreadable files, an unknown parameter); the reason goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .settings import Settings, load_settings
from .simulation import (
    SWEEPABLE,
    Trajectory,
    corridor_model,
    example_model,
    load_model,
    run_episode,
    save_table,
    summarize_sweep,
    sweep,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="YAML settings (default: ./config.yaml if any)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one setting, for example simulation.steps=50 (repeatable)",
    )


def _simulation_arguments(parser: argparse.ArgumentParser) -> None:
    _common(parser)
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--model", type=Path, help="YAML generative model")
    models.add_argument(
        "--builtin",
        choices=("two_state", "corridor"),
        default="two_state",
        help="built-in generative model (default: two_state)",
    )
    parser.add_argument("--output", type=Path, help="output directory (default: output.directory)")
    parser.add_argument("--no-figures", action="store_true", help="write tables and JSON only")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cognitive", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    show = commands.add_parser("settings", help="print the resolved settings")
    _common(show)
    show.add_argument("--json", action="store_true", help="print JSON instead of YAML-like text")

    simulate = commands.add_parser("simulate", help="run one episode")
    _simulation_arguments(simulate)
    simulate.add_argument("--steps", type=int, help="override simulation.steps")
    simulate.add_argument("--seed", type=int, help="override simulation.seed")

    sweeping = commands.add_parser("sweep", help="vary one inference parameter")
    _simulation_arguments(sweeping)
    sweeping.add_argument("--parameter", required=True, choices=sorted(SWEEPABLE))
    sweeping.add_argument("--values", required=True, help="comma-separated values to try")
    sweeping.add_argument("--seeds", type=int, default=5, help="episodes per value (default 5)")
    sweeping.add_argument("--steps", type=int, help="override simulation.steps")

    plotting = commands.add_parser("plot", help="redraw a saved discrete trajectory")
    _common(plotting)
    plotting.add_argument("trajectory", type=Path, help="saved trajectory.json")
    plotting.add_argument("--output", type=Path, help="output directory")
    plotting.add_argument("--step", type=int, default=0, help="step for expected free energy bars")
    plotting.add_argument("--json", action="store_true", help="print written files as JSON")

    for name in ("continuous", "hierarchical"):
        episode = commands.add_parser(name, help=f"run a {name} episode")
        _common(episode)
        episode.add_argument("--steps", type=int, help="number of world steps")
        episode.add_argument("--seed", type=int, help="random seed")
        episode.add_argument("--output", type=Path, help="output directory")
        episode.add_argument("--no-figures", action="store_true", help="write data only")
        episode.add_argument("--json", action="store_true", help="print output files as JSON")
        if name == "continuous":
            episode.add_argument("--dt", type=float, default=0.01, help="world time step")
            episode.add_argument("--alpha", type=float, default=0.1, help="belief update rate")
            episode.add_argument(
                "--sensor-noise",
                type=float,
                default=0.0,
                help="observation noise standard deviation",
            )
        else:
            episode.add_argument(
                "--decision-interval", type=int, default=5, help="world steps per upper decision"
            )

    commands.add_parser(
        "benchmark",
        help="time the runtime components (arguments after the command go to the benchmark)",
        add_help=False,
    )
    return parser


def _overrides(pairs: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--set expects SECTION.KEY=VALUE, got {pair!r}")
        parsed[key.strip()] = value.strip()
    return parsed


def _resolve(args: argparse.Namespace) -> Settings:
    overrides = _overrides(args.set)
    for option, key in (("steps", "simulation.steps"), ("seed", "simulation.seed")):
        value = getattr(args, option, None)
        if value is not None:
            overrides[key] = str(value)
    return load_settings(args.config, overrides=overrides)


def _model(args: argparse.Namespace) -> tuple[Any, dict[str, list[str]]]:
    if args.model:
        return load_model(args.model)
    if args.builtin == "corridor":
        return corridor_model(), {
            "states": [f"cell {i}" for i in range(4)],
            "observations": [f"cell {i}" for i in range(4)],
            "actions": ["left", "stay", "right"],
        }
    return example_model(), {
        "states": ["dark", "light"],
        "actions": ["stay", "flip"],
        "observations": ["dim", "bright"],
    }


def _output(args: argparse.Namespace, settings: Settings) -> Path:
    return args.output or Path(settings.output.directory)


def _use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _settings_command(args: argparse.Namespace) -> int:
    settings = _resolve(args)
    if args.json:
        print(json.dumps(settings.to_dict(), indent=2))
    else:
        for section, values in settings.to_dict().items():
            print(f"{section}:")
            for key, value in values.items():
                print(f"  {key}: {value}")
    return 0


def _simulate_command(args: argparse.Namespace) -> int:
    settings = _resolve(args)
    model, labels = _model(args)
    config = settings.inference_config()
    trajectory = run_episode(
        model, config, settings.simulation.steps, seed=settings.simulation.seed, labels=labels
    )
    out = _output(args, settings)
    written = [
        trajectory.save_json(out / "trajectory.json"),
        trajectory.save_csv(out / "trajectory.csv"),
    ]
    if not args.no_figures:
        _use_agg()
        from .visualization.trajectory_plots import (
            plot_efe_components,
            plot_model,
            plot_trajectory,
            save_figure,
        )

        suffix, dpi = settings.output.figure_format, settings.output.dpi
        written += [
            save_figure(plot_trajectory(trajectory), out / f"episode.{suffix}", dpi),
            save_figure(plot_efe_components(trajectory, 0), out / f"efe_step0.{suffix}", dpi),
            save_figure(plot_model(model), out / f"model.{suffix}", dpi),
        ]
    result = {"summary": trajectory.summary(), "files": [str(path) for path in written]}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result["summary"].items():
            print(f"{key}: {value}")
        for path in result["files"]:
            print(path)
    return 0


def _sweep_command(args: argparse.Namespace) -> int:
    settings = _resolve(args)
    model, _ = _model(args)
    if args.seeds < 1:
        raise ValueError("--seeds must be at least one")
    values: list[Any] = [item.strip() for item in args.values.split(",") if item.strip()]
    if not values:
        raise ValueError("--values must list at least one value")
    rows = sweep(
        model,
        settings.inference_config(),
        args.parameter,
        values,
        steps=settings.simulation.steps,
        seeds=range(settings.simulation.seed, settings.simulation.seed + args.seeds),
    )
    summary = summarize_sweep(rows)
    out = _output(args, settings)
    written = [
        save_table([row.flat() for row in rows], out / "sweep_episodes.csv"),
        save_table(summary, out / "sweep_summary.csv"),
    ]
    if not args.no_figures:
        _use_agg()
        from .visualization.trajectory_plots import plot_sweep, save_figure

        written.append(
            save_figure(
                plot_sweep(summary),
                out / f"sweep.{settings.output.figure_format}",
                settings.output.dpi,
            )
        )
    result = {"summary": summary, "files": [str(path) for path in written]}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for row in summary:
            print(
                f"{row['parameter']}={row['value']}: preference_rate={row['preference_rate']:.3f} "
                f"belief_accuracy={row['belief_accuracy']:.3f} "
                f"mean_free_energy={row['mean_free_energy']:.3f}"
            )
        for path in result["files"]:
            print(path)
    return 0


def _plot_command(args: argparse.Namespace) -> int:
    """Load recorded data and render it without running inference again."""
    settings = _resolve(args)
    trajectory = Trajectory.load_json(args.trajectory)
    if not 0 <= args.step < trajectory.steps:
        raise ValueError(f"step must be in [0, {trajectory.steps - 1}]")
    _use_agg()
    from .visualization.trajectory_plots import plot_efe_components, plot_trajectory, save_figure

    out = _output(args, settings)
    suffix, dpi = settings.output.figure_format, settings.output.dpi
    files = [
        save_figure(plot_trajectory(trajectory), out / f"episode.{suffix}", dpi),
        save_figure(
            plot_efe_components(trajectory, args.step), out / f"efe_step{args.step}.{suffix}", dpi
        ),
    ]
    paths = [str(path) for path in files]
    print(json.dumps({"files": paths}, indent=2) if args.json else "\n".join(paths))
    return 0


def _episode_command(args: argparse.Namespace) -> int:
    """Run the configured continuous or two-level corridor example."""
    from Things.Continuous_Generic import ContinuousActiveInference

    from .episodes import corridor_hierarchy, run_continuous_episode, run_hierarchical_episode

    settings = _resolve(args)
    out = _output(args, settings)
    if args.command == "continuous":
        continuous = run_continuous_episode(
            ContinuousActiveInference(dt=args.dt, alpha=args.alpha),
            settings.simulation.steps,
            seed=settings.simulation.seed,
            sensor_noise=args.sensor_noise,
        )
        written = [
            continuous.save_json(out / "trajectory.json"),
            continuous.save_csv(out / "trajectory.csv"),
        ]
        if not args.no_figures:
            _use_agg()
            from .visualization.trajectory_plots import plot_continuous_trajectory, save_figure

            written.append(
                save_figure(
                    plot_continuous_trajectory(continuous),
                    out / f"episode.{settings.output.figure_format}",
                    settings.output.dpi,
                )
            )
    else:
        hierarchical = run_hierarchical_episode(
            corridor_hierarchy(),
            settings.inference_config(),
            settings.inference_config(),
            settings.simulation.steps,
            seed=settings.simulation.seed,
            decision_interval=args.decision_interval,
        )
        written = [
            hierarchical.save_json(out / "hierarchy.json"),
            hierarchical.lower.save_json(out / "trajectory.json"),
            hierarchical.lower.save_csv(out / "trajectory.csv"),
        ]
        if not args.no_figures:
            _use_agg()
            from .visualization.trajectory_plots import plot_hierarchical_trajectory, save_figure

            written.append(
                save_figure(
                    plot_hierarchical_trajectory(hierarchical),
                    out / f"episode.{settings.output.figure_format}",
                    settings.output.dpi,
                )
            )
    paths = [str(path) for path in written]
    print(json.dumps({"files": paths}, indent=2) if args.json else "\n".join(paths))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["benchmark"]:
        from .benchmarks import main as benchmark_main

        return benchmark_main(arguments[1:])
    args = build_parser().parse_args(arguments)
    handlers = {
        "settings": _settings_command,
        "simulate": _simulate_command,
        "sweep": _sweep_command,
        "plot": _plot_command,
        "continuous": _episode_command,
        "hierarchical": _episode_command,
    }
    try:
        return handlers[args.command](args)
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
