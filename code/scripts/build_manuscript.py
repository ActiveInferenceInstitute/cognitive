"""Build the executable Cognitive Active Inference manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from cognitive import __version__
from cognitive.models.active_inference.base import ModelState
from cognitive.models.active_inference.dispatcher import ActiveInferenceDispatcher, InferenceConfig
from cognitive.models.active_inference.generative_model import DiscreteGenerativeModel
from Things.Continuous_Generic import ContinuousActiveInference

TOKEN_PATTERN = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Manuscript configuration must be a mapping")
    expected = {"paper", "authors", "publication", "keywords", "metadata", "render", "experiment"}
    unknown = set(config).difference(expected)
    if unknown:
        raise ValueError(f"Unknown manuscript configuration fields: {sorted(unknown)}")
    required = {"paper", "authors", "publication", "keywords", "metadata", "render", "experiment"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing manuscript configuration fields: {sorted(missing)}")
    section_fields = {
        "paper": {"title", "subtitle", "version", "date"},
        "publication": {"repository", "license"},
        "metadata": {"language", "audience"},
        "render": {"formats"},
    }
    section_required = {
        "paper": {"title", "version", "date"},
        "publication": {"repository", "license"},
        "metadata": {"language", "audience"},
        "render": {"formats"},
    }
    for section, allowed in section_fields.items():
        value = config[section]
        if not isinstance(value, dict):
            raise ValueError(f"{section} must be a mapping")
        unknown_section = set(value).difference(allowed)
        if unknown_section:
            raise ValueError(f"Unknown {section} fields: {sorted(unknown_section)}")
        missing_section = section_required[section].difference(value)
        if missing_section:
            raise ValueError(f"Missing {section} fields: {sorted(missing_section)}")
    authors = config["authors"]
    if not isinstance(authors, list) or not authors:
        raise ValueError("authors must be a non-empty list")
    for author in authors:
        if not isinstance(author, dict):
            raise ValueError("each author must be a mapping")
        unknown_author = set(author).difference({"name", "affiliation", "corresponding"})
        if unknown_author:
            raise ValueError(f"Unknown author fields: {sorted(unknown_author)}")
    keywords = config["keywords"]
    if not isinstance(keywords, list) or not all(isinstance(item, str) for item in keywords):
        raise ValueError("keywords must be a list of strings")
    formats = config["render"].get("formats")
    if not isinstance(formats, dict) or set(formats) != {"markdown", "html", "pdf"}:
        raise ValueError("render.formats must define markdown, html, and pdf")
    if not all(isinstance(value, bool) for value in formats.values()):
        raise ValueError("render.formats values must be boolean")
    experiment = config["experiment"]
    if not isinstance(experiment, dict):
        raise ValueError("experiment must be a mapping")
    experiment_keys = {"seed", "horizon", "observations", "model", "continuous_steps"}
    unknown_experiment = set(experiment).difference(experiment_keys)
    if unknown_experiment:
        raise ValueError(f"Unknown experiment fields: {sorted(unknown_experiment)}")
    if not isinstance(experiment.get("seed"), int) or experiment["seed"] < 0:
        raise ValueError("experiment.seed must be a non-negative integer")
    if not isinstance(experiment.get("horizon"), int) or experiment["horizon"] < 1:
        raise ValueError("experiment.horizon must be positive")
    observations = experiment.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("experiment.observations must be a non-empty list")
    if not isinstance(experiment.get("model"), dict):
        raise ValueError("experiment.model must be a mapping")
    continuous_steps = experiment.get("continuous_steps")
    if not isinstance(continuous_steps, int) or continuous_steps < 2:
        raise ValueError("experiment.continuous_steps must be at least two")
    return config


def _model_and_results(config: dict[str, Any]) -> tuple[DiscreteGenerativeModel, dict[str, Any]]:
    experiment = config["experiment"]
    model = DiscreteGenerativeModel.from_config(experiment["model"])
    observations = [int(value) for value in experiment["observations"]]
    if any(value < 0 or value >= model.num_observations for value in observations):
        raise ValueError("experiment.observations contains an out-of-range observation")
    methods = ("variational", "mean_field", "sampling")
    beliefs: dict[str, list[list[float]]] = {}
    policies: dict[str, list[float]] = {}
    state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
    for method in methods:
        dispatcher = ActiveInferenceDispatcher(
            InferenceConfig(
                method=method,
                policy_type="discrete",
                temporal_horizon=int(experiment["horizon"]),
                learning_rate=0.5,
                precision_init=1.0,
                num_samples=256,
                policy_limit=4096,
                seed=int(experiment["seed"]),
            ),
            model,
        )
        current = state.copy()
        trajectory = [current.beliefs.tolist()]
        for observation in observations:
            current.beliefs = dispatcher.dispatch_belief_update(observation, current)
            trajectory.append(current.beliefs.tolist())
        beliefs[method] = trajectory
        policies[method] = dispatcher.dispatch_policy_inference(current).tolist()

    efe = [model.expected_free_energy(model.D, action) for action in range(model.num_actions)]
    continuous = ContinuousActiveInference(
        n_states=model.num_states,
        n_obs=model.num_observations,
        n_orders=3,
        seed=int(experiment["seed"]),
    )
    continuous_time: list[float] = []
    continuous_beliefs: list[list[float]] = []
    for index in range(int(experiment["continuous_steps"])):
        observation_index = observations[index % len(observations)]
        observation = np.zeros(model.num_observations, dtype=float)
        observation[observation_index] = 1.0
        continuous_time.append(float(continuous.state.time))
        continuous_beliefs.append(continuous.state.belief_means[:, 0].tolist())
        continuous.step(observation)

    policy_count = model.num_actions ** int(experiment["horizon"])
    return model, {
        "methods": list(methods),
        "observations": observations,
        "beliefs": beliefs,
        "policies": policies,
        "efe": efe,
        "continuous_time": continuous_time,
        "continuous_beliefs": continuous_beliefs,
        "policy_count": policy_count,
    }


def _save_figures(
    output: Path, model: DiscreteGenerativeModel, results: dict[str, Any]
) -> list[dict[str, Any]]:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    registry: list[dict[str, Any]] = []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in results["methods"]:
        trajectory = np.asarray(results["beliefs"][method], dtype=float)
        ax.plot(trajectory[:, 0], label=f"{method}: state 0")
        if model.num_states > 1:
            ax.plot(trajectory[:, 1], linestyle="--", label=f"{method}: state 1")
    ax.set_xlabel("Observation update")
    ax.set_ylabel("Posterior belief")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Exact Bayesian posterior under three inference paths")
    ax.legend(fontsize="small", ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "belief_updates.png", dpi=180)
    plt.close(fig)
    registry.append(
        {
            "label": "fig:belief_updates",
            "file": "figures/belief_updates.png",
            "source": "build_manuscript.py",
        }
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(model.num_actions)
    width = 0.8 / len(results["methods"])
    for index, method in enumerate(results["methods"]):
        ax.bar(x + (index - 1) * width, results["policies"][method], width, label=method)
    ax.set_xlabel("First action")
    ax.set_ylabel("Policy probability")
    ax.set_xticks(x, [str(value) for value in x])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Normalized policy distributions")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "policy_distributions.png", dpi=180)
    plt.close(fig)
    registry.append(
        {
            "label": "fig:policy_distributions",
            "file": "figures/policy_distributions.png",
            "source": "build_manuscript.py",
        }
    )

    efe = np.asarray(results["efe"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, efe[:, 1], 0.2, label="risk")
    ax.bar(x, efe[:, 2], 0.2, label="ambiguity")
    ax.bar(x + 0.2, -efe[:, 3], 0.2, label="epistemic value")
    ax.set_xlabel("Action")
    ax.set_ylabel("Expected free-energy component")
    ax.set_xticks(x, [str(value) for value in x])
    ax.set_title("Risk, ambiguity, and epistemic value")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "efe_decomposition.png", dpi=180)
    plt.close(fig)
    registry.append(
        {
            "label": "fig:efe_decomposition",
            "file": "figures/efe_decomposition.png",
            "source": "build_manuscript.py",
        }
    )

    continuous_beliefs = np.asarray(results["continuous_beliefs"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for state in range(continuous_beliefs.shape[1]):
        ax.plot(results["continuous_time"], continuous_beliefs[:, state], label=f"state {state}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Generalized-coordinate mean")
    ax.set_title("Continuous precision-weighted belief dynamics")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "continuous_trajectory.png", dpi=180)
    plt.close(fig)
    registry.append(
        {
            "label": "fig:continuous_trajectory",
            "file": "figures/continuous_trajectory.png",
            "source": "build_manuscript.py",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    axes[0].imshow(model.A, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title("A: observation")
    axes[1].imshow(model.B[:, :, 0], cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("B: action 0")
    axes[2].imshow(model.B[:, :, min(1, model.num_actions - 1)], cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("B: action 1")
    for axis in axes:
        axis.set_xlabel("column")
        axis.set_ylabel("row")
    fig.suptitle("Validated generative-model matrices")
    fig.tight_layout()
    fig.savefig(figure_dir / "model_matrices.png", dpi=180)
    plt.close(fig)
    registry.append(
        {
            "label": "fig:model_matrices",
            "file": "figures/model_matrices.png",
            "source": "build_manuscript.py",
        }
    )
    return registry


def _variables(
    config: dict[str, Any],
    model: DiscreteGenerativeModel,
    results: dict[str, Any],
    figure_count: int,
) -> dict[str, str]:
    experiment = config["experiment"]
    return {
        "MODEL_STATES": str(model.num_states),
        "MODEL_OBSERVATIONS": str(model.num_observations),
        "MODEL_ACTIONS": str(model.num_actions),
        "MODEL_A_SHAPE": str(tuple(model.A.shape)),
        "MODEL_B_SHAPE": str(tuple(model.B.shape)),
        "HORIZON": str(experiment["horizon"]),
        "SEED": str(experiment["seed"]),
        "OBS_SEQUENCE": ", ".join(str(value) for value in results["observations"]),
        "POLICY_COUNT": str(results["policy_count"]),
        "METHOD_COUNT": str(len(results["methods"])),
        "FIGURE_COUNT": str(figure_count),
        "PACKAGE_VERSION": __version__,
    }


def _hydrate_sources(output: Path, variables: dict[str, str], manuscript_dir: Path) -> Path:
    sections = sorted(manuscript_dir.glob("[0-9][0-9]_*.md"))
    if not sections:
        raise ValueError(f"No numbered manuscript sections found in {manuscript_dir}")
    combined: list[str] = []
    for path in sections:
        content = path.read_text(encoding="utf-8")
        content = TOKEN_PATTERN.sub(lambda match: variables[match.group(1)], content)
        unresolved = TOKEN_PATTERN.findall(content)
        if unresolved:
            raise ValueError(
                f"Unresolved manuscript variables in {path}: {sorted(set(unresolved))}"
            )
        combined.append(content.rstrip())
    combined_path = output / "combined.md"
    combined_path.write_text("\n\n".join(combined) + "\n", encoding="utf-8")
    return combined_path


def _strip_crossref_caption_prefix(document: dict[str, Any]) -> None:
    """Remove prefixes that LaTeX adds again around crossref captions."""

    def strip_inlines(inlines: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(inlines) >= 4:
            first = inlines[0].get("c")
            number = inlines[2].get("c")
            if (
                inlines[0].get("t") == "Str"
                and first in {"Figure", "Table"}
                and inlines[1].get("t") == "Space"
                and inlines[2].get("t") == "Str"
                and isinstance(number, str)
                and number.rstrip(":").isdigit()
                and inlines[3].get("t") == "Space"
            ):
                return inlines[4:]
        return inlines

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("t") in {"Figure", "Table"}:
                caption = node.get("c", [None, None])[1]
                if isinstance(caption, list) and len(caption) == 2:
                    blocks = caption[1]
                    if blocks and blocks[0].get("t") == "Plain":
                        blocks[0]["c"] = strip_inlines(blocks[0]["c"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(document)


def _render(
    output: Path,
    combined: Path,
    config: dict[str, Any],
    manuscript_dir: Path,
    render_pdf: bool,
) -> None:
    paper = config["paper"]
    bibliography = output / "references.bib"
    preamble = output / "preamble.tex"
    preamble_source = (manuscript_dir / "preamble.md").read_text(encoding="utf-8")
    latex_blocks = re.findall(r"```latex\n(.*?)\n```", preamble_source, flags=re.DOTALL)
    preamble.write_text("\n\n".join(latex_blocks) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="cognitive-crossref-") as temporary_dir:
        crossref_json = Path(temporary_dir) / "crossref.json"
        preprocess = [
            "pandoc",
            str(combined),
            "--from=markdown+tex_math_dollars+raw_tex",
            "--filter",
            "pandoc-crossref",
            "--number-sections",
            "--quiet",
            "--to=json",
            "--output=" + str(crossref_json),
        ]
        subprocess.run(preprocess, check=True, cwd=output)
        crossref_document = json.loads(crossref_json.read_text(encoding="utf-8"))
        _strip_crossref_caption_prefix(crossref_document)
        crossref_json.write_text(json.dumps(crossref_document), encoding="utf-8")
        common = [
            "pandoc",
            str(crossref_json),
            "--from=json",
            "--citeproc",
            "--bibliography=" + str(bibliography),
            "--number-sections",
            "--resource-path=" + str(output),
            "--metadata",
            f"title={paper['title']}",
            "--metadata",
            f"subtitle={paper.get('subtitle', '')}",
            "--metadata",
            f"date={paper.get('date', '')}",
            "--quiet",
        ]
        html_command = common + [
            "--standalone",
            "--to=html5",
            "--output=" + str(output / "manuscript.html"),
        ]
        subprocess.run(html_command, check=True, cwd=output)
        if render_pdf:
            pdf_command = common + [
                "--pdf-engine=xelatex",
                "--include-in-header=" + str(preamble),
                "--to=pdf",
                "--output=" + str(output / "manuscript.pdf"),
            ]
            subprocess.run(pdf_command, check=True, cwd=output)


def _is_build_output(path: Path) -> bool:
    """True when the path looks like an artifact produced by this builder."""
    return (path / "build_manifest.json").is_file() or (
        path / "figure_registry.json"
    ).is_file()


def _prepare_output(output: Path) -> None:
    """Create the output directory, refusing to silently delete non-build dirs.

    The builder replaces its own previous output (identified by the manifest it
    writes), but must never recursively delete a directory the caller pointed
    at by mistake (repo root, docs tree, a source directory).
    """
    if output.exists():
        if _is_build_output(output):
            shutil.rmtree(output)
        elif output == Path.cwd().resolve():
            raise ValueError(f"Refusing to delete the working directory: {output}")
        elif any(output.iterdir()):
            raise ValueError(
                f"Refusing to delete non-build directory {output}: it is not empty and "
                "contains no build_manifest.json / figure_registry.json. Choose an empty "
                "or previously generated output path (or a path under build/)."
            )
        else:
            output.rmdir()
    output.mkdir(parents=True)


def build(
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
    render_pdf: bool = True,
) -> Path:
    """Build figures, hydrated source, and rendered manuscript artifacts."""
    config_file = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else Path.cwd() / "docs" / "manuscript" / "config.yaml"
    )
    manuscript_dir = config_file.parent
    config = _load_config(config_file)
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else Path.cwd() / "build" / "manuscript"
    )
    _prepare_output(output)
    model, results = _model_and_results(config)
    registry = _save_figures(output, model, results)
    variables = _variables(config, model, results, len(registry))
    (output / "references.bib").write_text(
        (manuscript_dir / "references.bib").read_text(encoding="utf-8"), encoding="utf-8"
    )
    combined = _hydrate_sources(output, variables, manuscript_dir)
    _render(output, combined, config, manuscript_dir, render_pdf=render_pdf)
    manifest = {
        "schema_version": 1,
        "package_version": __version__,
        "config": str(config_file),
        "paper": config["paper"],
        "authors": config["authors"],
        "publication": config["publication"],
        "keywords": config["keywords"],
        "metadata": config["metadata"],
        "render": config["render"],
        "variables": variables,
        "figures": registry,
        "combined_sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
    }
    (output / "figure_registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    (output / "manuscript_variables.json").write_text(
        json.dumps(variables, indent=2) + "\n", encoding="utf-8"
    )
    (output / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the executable Cognitive manuscript")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-pdf", action="store_true", help="Build HTML and figures without LaTeX PDF"
    )
    args = parser.parse_args(argv)
    output = build(args.config, args.output, render_pdf=not args.no_pdf)
    print(f"Built manuscript artifacts in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
