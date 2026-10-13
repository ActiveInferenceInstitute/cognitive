"""The figures render, carry the data they claim to, and are written and closed."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from cognitive.settings import Settings
from cognitive.simulation import (
    Trajectory,
    corridor_model,
    example_model,
    run_episode,
    summarize_sweep,
    sweep,
)
from cognitive.visualization.trajectory_plots import (
    plot_efe_components,
    plot_model,
    plot_sweep,
    plot_trajectory,
    save_figure,
)


@pytest.fixture(scope="module")
def trajectory():
    labels = {"states": ["dark", "light"], "actions": ["stay", "flip"], "observations": ["a", "b"]}
    return run_episode(example_model(), Settings().inference_config(), 12, seed=2, labels=labels)


def test_trajectory_figure_has_four_panels_and_the_titles_that_name_them(trajectory) -> None:
    fig = plot_trajectory(trajectory)
    try:
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert titles == [
            "Beliefs q(s) over time",
            "Choices and observations",
            "Variational free energy per step",
            "Action posterior P(a)",
        ]
        belief_image = fig.axes[0].images[0].get_array()
        assert np.allclose(np.asarray(belief_image), trajectory.beliefs.T)
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["dark", "light"]
        assert "seed 2" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_unlabelled_episodes_get_generated_labels() -> None:
    bare = run_episode(corridor_model(), Settings().inference_config(), 6, seed=0)
    fig = plot_trajectory(bare)
    try:
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["s0", "s1", "s2", "s3"]
    finally:
        plt.close(fig)


def test_saved_observation_names_and_unseen_preferred_cell_are_preserved() -> None:
    recorded = run_episode(
        corridor_model(slip=0, sensor_noise=0),
        Settings().inference_config(),
        1,
        labels={
            "actions": ["left", "stay", "right"],
            "observations": ["cell0", "cell1", "cell2", "cell3"],
        },
    )
    fig = plot_trajectory(recorded)
    try:
        labels = [tick.get_text() for tick in fig.axes[1].get_yticklabels()]
        assert labels == ["left | cell0", "stay | cell1", "right | cell2", " | cell3"]
        assert fig.axes[1].lines[1].get_ydata()[0] == 3
    finally:
        plt.close(fig)


def test_sparse_saved_indices_render_as_bounded_categorical_ticks(tmp_path: Path) -> None:
    payload = run_episode(example_model(), Settings().inference_config(), 1).to_dict()
    payload["observations"] = [2**53 + 1]
    recorded = Trajectory.from_dict(payload)
    fig = plot_trajectory(recorded)
    labels = [tick.get_text() for tick in fig.axes[1].get_yticklabels()]
    assert len(labels) == 3 and labels[-1] == f" | o{2**53 + 1}"
    assert list(fig.axes[1].lines[0].get_ydata()) == [2]
    path = save_figure(fig, tmp_path / "sparse.png")
    assert path.stat().st_size > 0


def test_efe_figure_marks_the_chosen_action_and_plots_each_component(trajectory) -> None:
    fig = plot_efe_components(trajectory, step=3)
    try:
        ax = fig.axes[0]
        chosen = int(trajectory.actions[3])
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels[chosen].endswith("(chosen)") and labels[1 - chosen] in ("stay", "flip")
        heights = [patch.get_height() for patch in ax.patches]
        assert len(heights) == 4 * trajectory.efe.shape[1]
        assert heights[:2] == pytest.approx(list(trajectory.risk[3]))
    finally:
        plt.close(fig)
    with pytest.raises(ValueError, match="step must be in"):
        plot_efe_components(trajectory, step=12)
    with pytest.raises(ValueError, match="step must be in"):
        plot_efe_components(trajectory, step=-1)


def test_sweep_figure_for_numbers_and_for_categories() -> None:
    model, config = example_model(), Settings().inference_config()
    numeric = summarize_sweep(
        sweep(model, config, "temperature", [3.0, 0.5, 1.0], steps=5, seeds=[0, 1])
    )
    fig = plot_sweep(numeric)
    try:
        assert len(fig.axes) == 3
        x, _ = fig.axes[0].lines[0].get_data()
        assert list(x) == [0.5, 1.0, 3.0]  # sorted for drawing
        assert "2 episodes per value" in fig._suptitle.get_text()
    finally:
        plt.close(fig)
    categories = summarize_sweep(
        sweep(model, config, "policy_aggregation", ["marginal", "min"], steps=5)
    )
    fig = plot_sweep(categories, metrics=("preference_rate",))
    try:
        assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["marginal", "min"]
    finally:
        plt.close(fig)
    with pytest.raises(ValueError, match="summary must not be empty"):
        plot_sweep([])


@pytest.mark.parametrize("model", [example_model(), corridor_model()])
def test_model_figure_has_one_panel_per_matrix(model) -> None:
    fig = plot_model(model)
    try:
        assert len(fig.axes) == model.num_actions + 3
        assert np.allclose(fig.axes[0].images[0].get_array(), model.A)
    finally:
        plt.close(fig)


def test_save_figure_creates_folders_picks_the_format_and_closes(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    number = fig.number
    path = save_figure(fig, tmp_path / "deep" / "plot.svg", dpi=50)
    assert path.read_text(encoding="utf-8").lstrip().startswith(("<?xml", "<svg"))
    assert not plt.fignum_exists(number)


def test_save_figure_closes_the_figure_even_when_saving_fails(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    number = fig.number
    with pytest.raises(ValueError, match="format"):
        save_figure(fig, tmp_path / "plot.unknownformat")
    assert not plt.fignum_exists(number)
