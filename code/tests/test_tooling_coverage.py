"""Coverage for the repository's supporting tools: plots, graphs, CLIs.

These modules ship as installed entry points, so a break in them is a break in
the documented interface even though no scientific result depends on them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from cognitive import benchmarks
from cognitive.models.matrices.matrix_ops import MatrixLoader, MatrixOps, MatrixVisualizer
from cognitive.utils.create_node import NodeCreator
from cognitive.utils.create_node import main as create_node_main
from cognitive.utils.visualization import network_viz
from cognitive.visualization.matrix_plots import MatrixPlotter, NetworkPlotter, StateSpacePlotter


def test_matrix_plotter_writes_every_documented_plot(tmp_path: Path) -> None:
    """Each plotting entry point produces a non-empty file on disk."""
    plotter = MatrixPlotter(str(tmp_path), {"figure_size": (4, 3), "dpi": 60})
    heatmap = plotter.plot_heatmap(np.array([[0.6, 0.4], [0.3, 0.7]]), title="A")
    assert heatmap is not None
    beliefs = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.2, 0.3, 0.5]])
    axes = plotter.plot_belief_evolution(beliefs, state_labels=["a", "b", "c"])
    saved = plotter.save_figure(axes.get_figure(), "belief_evolution")
    assert saved.is_file() and saved.stat().st_size > 0

    stamped = plotter.save_figure(
        plotter.plot_bar(np.array([0.2, 0.8])).get_figure(), "bar", use_timestamp=True
    )
    assert stamped.is_file() and stamped.name.startswith("bar_")

    axes = plotter.plot_free_energy_landscape(
        lambda beliefs: float(np.sum(np.square(beliefs))), num_points=6
    )
    assert axes is not None


def test_state_space_and_network_plotters_save_named_files(tmp_path: Path) -> None:
    """The optional save paths are exercised, not just the in-memory figures."""
    state_plotter = StateSpacePlotter(save_dir=tmp_path)
    beliefs = np.array([[0.5, 0.5], [0.7, 0.3], [0.9, 0.1]])
    state_plotter.plot_belief_evolution(beliefs, "Beliefs", ["a", "b"], save_name="beliefs")
    assert (tmp_path / "beliefs.png").is_file()

    network_plotter = NetworkPlotter(save_dir=tmp_path)
    figure = network_plotter.plot_message_network_dynamics(
        {"a": ["b"], "b": ["c"], "c": []},
        [{"source": "a", "target": "b", "content": {"value": 1}}],
        save_name="messages",
    )
    assert figure is not None
    assert (tmp_path / "messages.png").is_file()


def _vault(root: Path) -> Path:
    """Build a small Obsidian-style vault with resolvable wiki links."""
    knowledge = root / "knowledge_base"
    (knowledge / "agents").mkdir(parents=True)
    (knowledge / "beliefs").mkdir(parents=True)
    (knowledge / "agents" / "scout.md").write_text(
        "---\ntitle: Scout\n---\n\nLinks to [[curiosity]] and [[curiosity|the same page]].\n",
        encoding="utf-8",
    )
    (knowledge / "beliefs" / "curiosity.md").write_text(
        "---\ntitle: Curiosity\n---\n\nNo outgoing links.\n", encoding="utf-8"
    )
    config = root / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {"knowledge_base": "knowledge_base", "templates": "templates"},
                "visualization": {"seed": 3},
            }
        ),
        encoding="utf-8",
    )
    return config


def test_network_visualizer_builds_and_renders_a_graph(tmp_path: Path) -> None:
    """Nodes, edges, colours, and the HTML export all come from real files."""
    config = _vault(tmp_path)
    visualizer = network_viz.NetworkVisualizer(config)
    graph = visualizer.build_network()
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert visualizer._get_node_colors(graph)
    # Layout is seeded, so two computations place nodes identically.
    first = visualizer._get_node_positions(graph)
    second = visualizer._get_node_positions(graph)
    assert first.keys() == second.keys()
    for node in first:
        assert np.allclose(np.asarray(first[node]), np.asarray(second[node]))

    destination = tmp_path / "out" / "network.html"
    figure = visualizer.visualize(destination, width=400, height=300)
    assert figure is not None
    assert destination.is_file() and destination.stat().st_size > 0

    with pytest.raises(ValueError, match="width and height must be positive"):
        visualizer.visualize(None, width=0)


def test_network_layout_handles_a_graph_past_the_dense_solver_limit(tmp_path: Path) -> None:
    """networkx needs scipy from 500 nodes; the whole vault is far larger."""
    import networkx as nx

    visualizer = network_viz.NetworkVisualizer(_vault(tmp_path))
    positions = visualizer._get_node_positions(nx.path_graph(600))
    assert len(positions) == 600
    assert all(np.isfinite(np.asarray(p)).all() for p in positions.values())


def test_network_visualizer_rejects_invalid_configuration(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"paths": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="paths.knowledge_base"):
        network_viz.NetworkVisualizer(bad)

    not_a_mapping = tmp_path / "list.yaml"
    not_a_mapping.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        network_viz.NetworkVisualizer(not_a_mapping)

    bad_visualization = tmp_path / "viz.yaml"
    bad_visualization.write_text(
        yaml.safe_dump({"paths": {"knowledge_base": "kb"}, "visualization": ["seed"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="visualization must be a mapping"):
        network_viz.NetworkVisualizer(bad_visualization)


def test_network_visualizer_cli_writes_its_output(tmp_path: Path) -> None:
    config = _vault(tmp_path)
    destination = tmp_path / "cli.html"
    assert network_viz.main(["--config", str(config), "--output", str(destination)]) == 0
    assert destination.is_file()


def test_node_creator_renders_a_template(tmp_path: Path) -> None:
    """Templates render, and a second write refuses to overwrite the first."""
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "belief_template.md").write_text(
        "---\nid: {{ belief_id }}\ntype: {{ type }}\n---\n\n# {{ belief_name }}\n\n{{ summary }}\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"paths": {"templates": "templates", "knowledge_base": "kb"}}),
        encoding="utf-8",
    )
    creator = NodeCreator(config)
    created = creator.create_node("belief", "curiosity", {"summary": "A short note."})
    assert created.is_file()
    body = created.read_text(encoding="utf-8")
    assert "A short note." in body
    assert "# curiosity" in body
    assert created.parent.name == "beliefs"

    with pytest.raises(FileExistsError):
        creator.create_node("belief", "curiosity")
    # An unmapped type falls back to a simple plural directory.
    (templates / "insight_template.md").write_text("# {{ node_name }}\n", encoding="utf-8")
    assert creator.create_node("insight", "one").parent.name == "insights"
    with pytest.raises(FileNotFoundError, match="No template for node type"):
        creator.create_node("absent", "x")


def test_node_creator_rejects_unsafe_names_and_types(tmp_path: Path) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "belief_template.md").write_text("# {{ node_name }}\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"paths": {"templates": "templates", "knowledge_base": "kb"}}),
        encoding="utf-8",
    )
    creator = NodeCreator(config)
    for unsafe in ("../escape", "a/b", "..", "   ", "///"):
        with pytest.raises(ValueError):
            creator.create_node("belief", unsafe)
    for bad_type in ("1belief", "be lief", "belief!"):
        with pytest.raises(ValueError, match="node_type must contain"):
            creator.create_node(bad_type, "safe")

    missing = tmp_path / "absent.yaml"
    with pytest.raises(ValueError, match="Unable to read configuration"):
        NodeCreator(missing)
    no_paths = tmp_path / "no_paths.yaml"
    no_paths.write_text(yaml.safe_dump({"other": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="paths mapping"):
        NodeCreator(no_paths)
    empty_path = tmp_path / "empty_path.yaml"
    empty_path.write_text(yaml.safe_dump({"paths": {"templates": "", "knowledge_base": "kb"}}))
    with pytest.raises(ValueError, match="must be a non-empty path"):
        NodeCreator(empty_path)


def test_node_creator_cli_reports_errors_without_a_traceback(tmp_path: Path) -> None:
    from click.testing import CliRunner

    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "goal_template.md").write_text(
        "# {{ node_name }}\n{{ detail }}\n", encoding="utf-8"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"paths": {"templates": "templates", "knowledge_base": "kb"}}),
        encoding="utf-8",
    )
    runner = CliRunner()
    ok = runner.invoke(
        create_node_main,
        ["--type", "goal", "--name", "ship", "--config", str(config), "--var", "detail=soon"],
    )
    assert ok.exit_code == 0, ok.output
    assert (tmp_path / "kb" / "goals" / "ship.md").is_file()

    malformed = runner.invoke(
        create_node_main,
        ["--type", "goal", "--name", "other", "--config", str(config), "--var", "novalue"],
    )
    assert malformed.exit_code != 0
    assert "key=value" in malformed.output

    empty_key = runner.invoke(
        create_node_main,
        ["--type", "goal", "--name", "other", "--config", str(config), "--var", "=v"],
    )
    assert empty_key.exit_code != 0

    duplicate = runner.invoke(
        create_node_main,
        ["--type", "goal", "--name", "ship", "--config", str(config), "--var", "detail=again"],
    )
    assert duplicate.exit_code != 0
    assert "already exists" in duplicate.output


def test_matrix_loader_and_visualizer_adapters(tmp_path: Path) -> None:
    """Specification loading, validation, and the plotting adapters."""
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("No frontmatter here.\n", encoding="utf-8")
    assert MatrixLoader.load_spec(spec_file) == {}

    spec_file.write_text("---\ndimensions:\n  shape: [2, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frontmatter terminator"):
        MatrixLoader.load_spec(spec_file)

    spec_file.write_text(
        "---\ndimensions:\n  shape: [2, 2]\nshape_constraints:\n"
        '  - "sum(cols) == 1.0"\n  - "all_values >= 0"\n---\n\nBody\n',
        encoding="utf-8",
    )
    spec = MatrixLoader.load_spec(spec_file)
    assert spec["dimensions"]["shape"] == [2, 2]

    data = tmp_path / "matrix.npy"
    np.save(data, np.array([[0.6, 0.4], [0.4, 0.6]]))
    matrix = MatrixLoader.load_matrix(data)
    assert MatrixLoader.validate_matrix(matrix, spec)
    assert not MatrixLoader.validate_matrix(np.array([[np.inf, 0.0], [0.0, 1.0]]), spec)
    assert not MatrixLoader.validate_matrix(np.eye(3), spec)
    assert not MatrixLoader.validate_matrix(np.array([[-1.0, 2.0], [2.0, -1.0]]), spec)
    row_spec = {"dimensions": {"rows": 2, "cols": 2}, "shape_constraints": ["sum(rows) == 1.0"]}
    assert MatrixLoader.validate_matrix(np.array([[0.5, 0.5], [0.2, 0.8]]), row_spec)
    assert not MatrixLoader.validate_matrix(np.array([[0.5, 0.9], [0.2, 0.8]]), row_spec)
    with pytest.raises(ValueError, match="Unsupported matrix constraints"):
        MatrixLoader.validate_matrix(matrix, {"shape_constraints": ["sum(diagonal) == 1.0"]})

    heatmap = MatrixVisualizer.prepare_heatmap_data(np.eye(2))
    assert list(heatmap["x_ticks"]) == [0, 1]
    assert list(MatrixVisualizer.prepare_bar_data(np.array([1.0, 2.0]))["x_ticks"]) == [0, 1]
    multi = MatrixVisualizer.prepare_multi_heatmap_data(np.zeros((3, 2, 2)))
    assert len(multi["slices"]) == 3
    with pytest.raises(ValueError, match="two-dimensional"):
        MatrixVisualizer.prepare_heatmap_data(np.zeros(2))
    with pytest.raises(ValueError, match="three-dimensional"):
        MatrixVisualizer.prepare_multi_heatmap_data(np.zeros((2, 2)))


def test_matrix_ops_guards_are_exercised() -> None:
    """The validation branches of the shared numerical helpers."""
    assert MatrixOps.normalize_columns(np.zeros((2, 2))) == pytest.approx(np.full((2, 2), 0.5))
    assert MatrixOps.ensure_probability_distribution(
        np.array([[1.0], [3.0]])
    ).ravel() == pytest.approx([0.25, 0.75])
    with pytest.raises(ValueError, match="two-dimensional"):
        MatrixOps.normalize_rows(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="non-empty finite"):
        MatrixOps.normalize_columns(np.array([np.nan, 1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        MatrixOps.compute_entropy(np.array([-0.5, 1.5]))
    with pytest.raises(ValueError, match="same shape"):
        MatrixOps.compute_kl_divergence(np.array([0.5, 0.5]), np.array([1.0]))
    with pytest.raises(ValueError, match="temperature"):
        MatrixOps.softmax(np.zeros(2), temperature=-1.0)
    axis_entropy = MatrixOps.compute_entropy(np.array([[0.5, 0.5], [1.0, 0.0]]), axis=1)
    assert axis_entropy == pytest.approx([np.log(2.0), 0.0])


def test_benchmarks_report_is_machine_readable(tmp_path: Path) -> None:
    """The benchmark CLI writes a schema-tagged report with per-call timings."""
    destination = tmp_path / "nested" / "benchmarks.json"
    assert benchmarks.main(["--repetitions", "1", "--output", str(destination)]) == 0
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["schema_version"] >= 1
    assert report["repetitions"] == 1
    names = {entry["name"] for entry in report["benchmarks"]}
    assert {"matrix_operations", "dispatcher_inference", "simple_pomdp"} <= names
    for entry in report["benchmarks"]:
        assert entry["seconds_per_call"] > 0
        assert entry["total_seconds"] > 0
    with pytest.raises(ValueError, match="repetitions must be positive"):
        benchmarks.run_benchmarks(0)
