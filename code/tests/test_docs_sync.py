"""The reference pages for the new features name everything the code exposes.

A page that lists options by hand drifts the day an option is added. These
tests read the code (settings fields, argparse help, public plot functions,
example scripts) and require each name to appear on the page that documents it.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest
from cognitive import settings
from cognitive.cli import main as cli_main
from cognitive.visualization import trajectory_plots

REPO_ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def help_text(*arguments: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), pytest.raises(SystemExit):
        cli_main([*arguments, "--help"])
    return buffer.getvalue()


@pytest.mark.parametrize("section", ["active_inference", "simulation", "output"])
def test_settings_reference_documents_every_key_with_its_default(section: str) -> None:
    page = read("docs/config/settings.md")
    defaults = settings.Settings().to_dict()[section]
    for key in settings.section_keys(section):
        assert f"`{section}.{key}`" in page, f"settings.md does not document {section}.{key}"
        assert f"`{defaults[key]}`" in page or f"`{defaults[key]!r}`" in page, (
            f"settings.md does not state the default of {section}.{key} ({defaults[key]!r})"
        )


def test_settings_reference_states_the_environment_prefix_and_the_default_file() -> None:
    page = read("docs/config/settings.md")
    assert settings.ENV_PREFIX in page and settings.DEFAULT_CONFIG_NAME in page


@pytest.mark.parametrize("command", ["settings", "simulate", "sweep"])
def test_cli_reference_lists_every_subcommand_option(command: str) -> None:
    page = read("docs/cli/reference.md")
    assert f"cognitive {command}" in page
    for option in sorted(set(re.findall(r"--[a-z][a-z-]*", help_text(command)))):
        if option == "--help":
            continue
        assert option in page, f"docs/cli/reference.md does not mention {option} of {command}"


def test_cli_reference_covers_benchmark_and_the_report_and_export_commands() -> None:
    page = read("docs/cli/reference.md")
    for name in (
        "cognitive benchmark",
        "cognitive-kb-report",
        "cognitive-export-rdf",
        "python -m cognitive",
    ):
        assert name in page


def test_visualization_catalogue_names_every_public_plot_and_every_figure_file() -> None:
    page = read("docs/visualization/gallery.md")
    functions = [
        name
        for name in dir(trajectory_plots)
        if name.startswith(("plot_", "save_")) and callable(getattr(trajectory_plots, name))
    ]
    assert functions
    for name in functions:
        assert f"`{name}`" in page, f"gallery.md does not describe {name}"
    for image in re.findall(r"\]\((figures/[^)]+\.png)\)", page):
        assert (REPO_ROOT / "docs/visualization" / image).is_file(), image
    for name in ("in_degree_distribution", "domain_sizes", "domain_links", "relations"):
        assert name in page, f"gallery.md does not describe the kb_report figure {name}"


def test_examples_readme_lists_every_script_and_config() -> None:
    page = read("examples/README.md")
    for path in sorted((REPO_ROOT / "examples").glob("[0-9][0-9]_*.py")):
        assert f"`{path.name}`" in page, f"examples/README.md does not list {path.name}"
    for path in sorted((REPO_ROOT / "examples" / "configs").glob("*.yaml")):
        assert path.name in page, f"examples/README.md does not mention {path.name}"


def test_the_documentation_hubs_link_to_the_new_sections() -> None:
    hub = read("docs/README.md")
    for section in ("cli/", "visualization/", "architecture/", "config/settings.md"):
        assert section in hub, f"docs/README.md does not link {section}"
    assert "examples/README.md" in read("README.md")
