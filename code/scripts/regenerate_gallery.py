"""Regenerate the complete gallery from recorded episodes and the RDF page graph."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

from cognitive.episodes import corridor_hierarchy, run_continuous_episode, run_hierarchical_episode
from cognitive.settings import Settings
from cognitive.simulation import example_model, run_episode, save_table, summarize_sweep, sweep
from Things.Continuous_Generic import ContinuousActiveInference

from .kb_report import compute_report, draw_figures, knowledge_graph

FIGURES = {
    "episode.png": ("plot_trajectory", "Beliefs, true states, actions and free energy."),
    "efe_components.png": ("plot_efe_components", "Expected free energy parts at step 0."),
    "temperature_sweep.png": (
        "plot_sweep",
        "Means and population standard deviations across seeds.",
    ),
    "model.png": ("plot_model", "Likelihood, transition, preference and prior matrices."),
    "continuous.png": (
        "plot_continuous_trajectory",
        "Continuous states, belief means and residual actions.",
    ),
    "hierarchical.png": (
        "plot_hierarchical_trajectory",
        "Upper beliefs and goals against lower states and actions.",
    ),
    "kb_in_degree.png": (
        "in_degree_distribution",
        "Share of pages with at least k incoming links.",
    ),
    "kb_domain_sizes.png": (
        "domain_sizes",
        "Pages per domain and the share of links staying inside.",
    ),
    "kb_domain_links.png": (
        "domain_links",
        "Links from row domain to column domain, colored by log(1 + links).",
    ),
    "kb_relations.png": (
        "relations",
        "The twenty most frequent edge relation names, on a log scale.",
    ),
}


def gallery_markdown(parameters: dict, report: dict) -> str:
    """Build captions directly from the run parameters and measured graph report."""
    sections = [
        "---",
        "title: Figure gallery",
        "type: documentation",
        "status: stable",
        "updated: 2026-10-13",
        "semantic_relations:",
        "  - type: documents",
        "    links:",
        '      - "[[AGENTS|Visualization instructions]]"',
        '      - "[[README|Visualization]]"',
        "---",
        "",
        "# Figure gallery",
        "",
        "Every `plot_*` function returns a Matplotlib figure and leaves the backend alone. "
        "`save_figure` writes and closes a figure. Captions and figures are regenerated together.",
        "",
        "```bash",
        "uv run python -m scripts.regenerate_gallery --publish",
        "```",
        "",
        "The command writes recorded data and a hash receipt under `output/gallery`, then copies "
        "the figures and this page into the documentation with `--publish`. Without that option, "
        "it writes only under the chosen `--output` directory.",
        "",
        "## Reproduction parameters",
        "",
        "```json",
        json.dumps(parameters, indent=2),
        "```",
        "",
        f"The graph contains {report['pages']} pages, {report['links']} links, "
        f"{report['weak_components']} weak component(s) and "
        f"{len(report['isolated_pages'])} isolated page(s).",
        "",
        "The graph measurements describe documentation connectivity. They do not measure "
        "scientific validity or inferential performance.",
        "",
    ]
    for filename, (function, description) in FIGURES.items():
        sections.extend(
            [f"## `{function}`", "", f"![{description}](figures/{filename})", "", description, ""]
        )
        if filename in ("episode.png", "efe_components.png", "model.png"):
            sections.extend(
                [f"Two-state episode: {parameters['steps']} steps, seed {parameters['seed']}.", ""]
            )
        elif filename == "temperature_sweep.png":
            sections.extend(
                [
                    f"Temperatures {parameters['temperatures']}; "
                    f"{parameters['seeds']} seeds per value, "
                    f"{parameters['steps']} steps per episode. "
                    "The band is seed spread, not a confidence interval.",
                    "",
                ]
            )
        elif filename == "continuous.png":
            sections.extend(
                [
                    "The generalized-coordinate agent uses residual control in an Euler world; "
                    "it does not optimize expected free energy over continuous actions.",
                    "",
                ]
            )
        elif filename == "hierarchical.png":
            sections.extend(
                [
                    "The upper controller selects lower preferences every five world steps. "
                    "Lower beliefs and physical state persist between decisions.",
                    "",
                ]
            )
    sections.extend(
        [
            "## Saved-data plots",
            "",
            "`cognitive plot trajectory.json --step 0` redraws a validated "
            "discrete data without rerunning inference. `plot_model` requires the original model.",
            "",
            "`cognitive-kb-report` writes the four knowledge-base figures separately.",
            "",
        ]
    )
    return "\n".join(sections)


def regenerate_gallery(
    root: Path, output: Path, *, steps: int = 30, seed: int = 7, seeds: int = 10
) -> dict:
    """Record source hashes, generate all ten figures and write matching captions."""
    if min(steps, seeds) < 1 or seed < 0:
        raise ValueError("steps and seeds must be positive; seed must be non-negative")
    graph = knowledge_graph(root)
    report = compute_report(graph)
    import matplotlib

    matplotlib.use("Agg")
    from cognitive.visualization.trajectory_plots import (
        plot_continuous_trajectory,
        plot_efe_components,
        plot_hierarchical_trajectory,
        plot_model,
        plot_sweep,
        plot_trajectory,
        save_figure,
    )

    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    config = Settings().inference_config()
    model = example_model()
    trajectory = run_episode(model, config, steps, seed=seed)
    trajectory.save_json(output / "trajectory.json")
    temperatures = [0.1, 0.3, 1.0, 3.0]
    summary = summarize_sweep(
        sweep(model, config, "temperature", temperatures, steps=steps, seeds=range(seeds))
    )
    save_table(summary, output / "sweep_summary.csv")
    continuous = run_continuous_episode(
        ContinuousActiveInference(), 100, seed=seed, sensor_noise=0.05
    )
    continuous.save_json(output / "continuous.json")
    hierarchical = run_hierarchical_episode(corridor_hierarchy(), config, config, steps, seed=seed)
    hierarchical.save_json(output / "hierarchy.json")
    for name, figure in (
        ("episode.png", plot_trajectory(trajectory)),
        ("efe_components.png", plot_efe_components(trajectory)),
        ("temperature_sweep.png", plot_sweep(summary)),
        ("model.png", plot_model(model)),
        ("continuous.png", plot_continuous_trajectory(continuous)),
        ("hierarchical.png", plot_hierarchical_trajectory(hierarchical)),
    ):
        save_figure(figure, figures / name)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for path in draw_figures(graph, report, output / "kb", image_format="png", dpi=150):
        target = {
            "in_degree_distribution": "kb_in_degree",
            "domain_sizes": "kb_domain_sizes",
            "domain_links": "kb_domain_links",
            "relations": "kb_relations",
        }[path.stem]
        shutil.copyfile(path, figures / f"{target}.png")
    parameters = {
        "steps": steps,
        "seed": seed,
        "temperatures": temperatures,
        "seeds": seeds,
        "continuous_steps": 100,
        "continuous_sensor_noise": 0.05,
        "decision_interval": 5,
    }
    (output / "gallery.md").write_text(gallery_markdown(parameters, report), encoding="utf-8")
    package_root = Path(inspect.getfile(Settings)).parent
    source_files = {
        f"code/tools/src/{path.relative_to(package_root)}": path
        for path in package_root.rglob("*.py")
    }
    source_files.update(
        {f"code/scripts/{path.name}": path for path in Path(__file__).parent.glob("*.py")}
    )
    continuous_root = Path(inspect.getfile(ContinuousActiveInference)).parent
    source_files.update(
        {
            f"code/Things/Continuous_Generic/{path.name}": path
            for path in continuous_root.glob("*.py")
        }
    )
    source_files["code/Things/__init__.py"] = continuous_root.parent / "__init__.py"
    source_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files.items()
    }
    source_hashes.update(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "knowledge_base").rglob("*.md"))
        }
    )
    receipt = {
        "schema_version": 1,
        "parameters": parameters,
        "runtime": {
            "python": sys.version,
            **{
                name: version(name)
                for name in (
                    "cognitive-active-inference",
                    "numpy",
                    "matplotlib",
                    "networkx",
                    "PyYAML",
                )
            },
        },
        "sources": source_hashes,
        "outputs": {
            str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "receipt.json"
        },
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Generate a gallery; publish only when explicitly requested on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("output/gallery"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--publish", action="store_true", help="update committed documentation and figures"
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve()
    receipt = regenerate_gallery(root, output, steps=args.steps, seed=args.seed, seeds=args.seeds)
    if args.publish:
        destination = root / "docs/visualization"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "gallery.md", destination / "gallery.md")
        shutil.copytree(output / "figures", destination / "figures", dirs_exist_ok=True)
    print(
        json.dumps(
            {
                "figures": len(FIGURES),
                "receipt": str(output / "receipt.json"),
                "parameters": receipt["parameters"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
