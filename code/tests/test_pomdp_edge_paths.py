"""Edge paths in SimplePOMDP matrix construction, validation, and plotting.

Every assertion here exercises a branch the main behavioural tests leave
uncovered: degenerate shapes, alternative constraints, invalid matrices, and
the plotting branches that only a history or a resized belief grid reaches.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from Things.Simple_POMDP.simple_pomdp import (
    SimplePOMDP,
    _normalize,
    _SimplePOMDPPlotter,
    _softmax,
)


def _base_config() -> dict[str, object]:
    return {
        "model": {"name": "edge", "description": "edge paths", "version": "0.1"},
        "state_space": {
            "num_states": 3,
            "state_labels": ["S1", "S2", "S3"],
            "initial_state": 0,
        },
        "observation_space": {
            "num_observations": 2,
            "observation_labels": ["O1", "O2"],
        },
        "action_space": {"num_actions": 2, "action_labels": ["A1", "A2"]},
        "matrices": {
            "A_matrix": {
                "shape": [2, 3],
                "initialization": "uniform",
                "constraints": ["column_stochastic", "non_negative"],
            },
            "B_matrix": {
                "shape": [3, 3, 2],
                "initialization": "identity_based",
                "initialization_params": {"strength": 0.8},
            },
            "C_matrix": {
                "shape": [2],
                "initialization": "log_preferences",
                "initialization_params": {"preferences": [0.1, 2.0]},
            },
            "D_matrix": {"shape": [3], "initialization": "uniform"},
            "E_matrix": {"shape": [2], "initialization": "uniform"},
        },
        "inference": {"time_horizon": 1, "learning_rate": 0.5},
        # Relative, so SimplePOMDP anchors it to the configuration file's
        # directory, which the caller places under tmp_path. A hardcoded
        # absolute path writes outside the test's sandbox and fails wherever
        # that location is not writable.
        "visualization": {"output_dir": "output", "formats": ["png"]},
    }


def _model_with(tmp_path: Path, mutate) -> SimplePOMDP:
    config = _base_config()
    mutate(config)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return SimplePOMDP(path)


def test_normalize_rejects_empty_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite and non-empty"):
        _normalize(np.array([]))
    with pytest.raises(ValueError, match="finite and non-empty"):
        _normalize(np.array([np.nan, 0.5]))


def test_normalize_falls_back_to_uniform_on_zero_mass() -> None:
    assert _normalize(np.zeros(4)) == pytest.approx(np.full(4, 0.25))


def test_softmax_validates_temperature() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _softmax(np.zeros(3), temperature=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        _softmax(np.zeros(3), temperature=float("inf"))
    # Extreme logits stay finite because of the max-shift.
    assert np.all(np.isfinite(_softmax(np.array([-1000.0, 0.0, 1000.0]))))


def test_identity_based_single_row_and_zero_strength(tmp_path: Path) -> None:
    """A one-row matrix is uniform, and strength 0 flattens everything."""
    single_row = SimplePOMDP._identity_based(SimplePOMDP, (1, 4), strength=0.9)
    assert single_row == pytest.approx(np.ones((1, 4)))
    # With strength 0 the target cell gets no extra mass, so every column is
    # uniform except that the zero diagonal entry stays zero.
    zero_strength = SimplePOMDP._identity_based(SimplePOMDP, (3, 3), strength=0.0)
    assert zero_strength == pytest.approx(0.5 * (np.ones((3, 3)) - np.eye(3)))
    flat_tensor = SimplePOMDP._identity_based(SimplePOMDP, (3, 3, 2), strength=0.0)
    assert flat_tensor.sum(axis=0) == pytest.approx(np.ones((3, 2)))
    with pytest.raises(ValueError, match="2D or 3D shapes"):
        SimplePOMDP._identity_based(SimplePOMDP, (4,), strength=0.9)


def test_constraints_reject_and_normalize(tmp_path: Path) -> None:
    model = _model_with(tmp_path, lambda c: None)
    with pytest.raises(ValueError, match="Unsupported matrix constraint"):
        SimplePOMDP._apply_constraint(model, np.ones((2, 2)), "bogus")
    with pytest.raises(ValueError, match="row_stochastic"):
        SimplePOMDP._apply_constraint(model, np.ones((2, 2, 2)), "row_stochastic")
    with pytest.raises(ValueError, match="all-zero"):
        SimplePOMDP._apply_constraint(model, np.zeros((2, 2)), "sum_to_one")
    row_normalized = SimplePOMDP._apply_constraint(
        model, np.array([[1.0, 3.0], [1.0, 1.0]]), "row_stochastic"
    )
    assert row_normalized.sum(axis=1) == pytest.approx([1.0, 1.0])
    summed = SimplePOMDP._apply_constraint(model, np.array([[2.0, 4.0]]), "sum_to_one")
    assert summed.sum() == pytest.approx(1.0)
    # _row_normalize divides by a guarded sum, so an all-zero row stays zero
    # rather than becoming NaN.
    guarded = SimplePOMDP._row_normalize(np.array([[0.0, 0.0], [1.0, 3.0]]))
    assert guarded[0] == pytest.approx([0.0, 0.0])
    assert guarded[1] == pytest.approx([0.25, 0.75])


def test_invalid_matrices_are_rejected(tmp_path: Path) -> None:
    def set_a(c: dict) -> None:
        c["matrices"]["A_matrix"] = {
            "shape": [3, 3],
            "initialization": "uniform",
        }

    with pytest.raises(ValueError, match="A matrix shape must match"):
        _model_with(tmp_path, set_a)

    def bad_c(c: dict) -> None:
        c["matrices"]["C_matrix"] = {
            "shape": [3],
            "initialization": "log_preferences",
            "initialization_params": {"preferences": [0.0, 1.0, 2.0]},
        }

    with pytest.raises(ValueError, match="C matrix"):
        _model_with(tmp_path, bad_c)

    def bad_d(c: dict) -> None:
        c["matrices"]["D_matrix"] = {"shape": [2], "initialization": "uniform"}

    with pytest.raises(ValueError, match="D matrix"):
        _model_with(tmp_path, bad_d)

    def bad_e(c: dict) -> None:
        c["matrices"]["E_matrix"] = {"shape": [3], "initialization": "uniform"}

    with pytest.raises(ValueError, match="E matrix"):
        _model_with(tmp_path, bad_e)

    def bad_initial(c: dict) -> None:
        c["state_space"]["initial_state"] = 9

    with pytest.raises(ValueError, match="initial_state"):
        _model_with(tmp_path, bad_initial)

    def bad_b(c: dict) -> None:
        c["matrices"]["B_matrix"] = {
            "shape": [3, 2, 2],
            "initialization": "identity_based",
        }

    with pytest.raises(ValueError, match="B matrix shape"):
        _model_with(tmp_path, bad_b)


def test_visualizer_without_history_uses_current_beliefs(tmp_path: Path) -> None:
    model = _model_with(tmp_path, lambda c: None)
    visualizer = _SimplePOMDPPlotter(model)
    fig = visualizer.visualize("belief_evolution")
    plt.close(fig)
    # After a run the history branch is taken instead.
    model.run(3)
    fig = visualizer.visualize("belief_evolution")
    plt.close(fig)


def test_belief_grid_resizes_when_belief_dimension_differs(tmp_path: Path) -> None:
    """The free-energy landscape resizes a 3-simplex grid onto 2 states."""
    model = _model_with(
        tmp_path,
        lambda c: (
            c["state_space"].update(num_states=2, state_labels=["S1", "S2"], initial_state=0),
            c["matrices"].update(
                A_matrix={
                    "shape": [2, 2],
                    "initialization": "uniform",
                    "constraints": ["column_stochastic"],
                },
                B_matrix={
                    "shape": [2, 2, 2],
                    "initialization": "identity_based",
                    "initialization_params": {"strength": 0.8},
                },
                D_matrix={"shape": [2], "initialization": "uniform"},
            ),
        ),
    )
    visualizer = _SimplePOMDPPlotter(model)
    fig = visualizer.visualize("free_energy_landscape")
    plt.close(fig)


def test_identity_based_strength_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="identity strength"):
        SimplePOMDP._identity_based(SimplePOMDP, (3, 3), strength=1.5)
    with pytest.raises(ValueError, match="identity strength"):
        SimplePOMDP._identity_based(SimplePOMDP, (3, 3), strength=-0.1)


def test_identity_based_partial_strength_column_forms() -> None:
    """A 0.6-strength 3x3 matrix keeps 0.6 on the diagonal cell per column."""
    matrix = SimplePOMDP._identity_based(SimplePOMDP, (3, 3), strength=0.6)
    assert matrix[0, 0] == pytest.approx(0.6)
    assert matrix[1, 0] == pytest.approx(0.2)
    assert matrix.sum(axis=0) == pytest.approx(np.ones(3))
    tensor = SimplePOMDP._identity_based(SimplePOMDP, (3, 3, 2), strength=0.6)
    # Action 1 shifts the target cell by one state.
    assert tensor[1, 0, 1] == pytest.approx(0.6)
    assert tensor[0, 0, 1] == pytest.approx(0.2)


# --- settings that used to be accepted and ignored ----------------------------


def test_a_per_matrix_learning_rate_is_rejected(tmp_path: Path) -> None:
    def mutate(config):
        config["matrices"]["E_matrix"]["learning_rate"] = 0.2

    with pytest.raises(ValueError, match="E_matrix.learning_rate is not used"):
        _model_with(tmp_path, mutate)


def test_plot_formats_dpi_and_style_are_applied(tmp_path: Path) -> None:
    def mutate(config):
        config["visualization"] = {
            "output_dir": "output",
            "formats": ["png", "svg"],
            "dpi": 50,
            "style": {"colormap": "magma", "font_size": 14},
        }

    model = _model_with(tmp_path, mutate)
    figure = model.visualize("observation_likelihood")
    output = tmp_path / "output"
    assert sorted(path.name for path in output.iterdir()) == [
        "observation_likelihood.png",
        "observation_likelihood.svg",
    ]
    # 6 x 4 inches at 50 dpi, plus the tight bounding box, is far below the
    # 600 x 400 pixels of the default.
    from matplotlib.image import imread

    assert imread(output / "observation_likelihood.png").shape[1] < 400
    assert figure is not None


@pytest.mark.parametrize(
    ("visualization", "message"),
    [
        ({"formats": []}, "formats must be a non-empty list"),
        ({"formats": ["gif"]}, "formats must be a non-empty list"),
        ({"formats": "png"}, "formats must be a non-empty list"),
        ({"dpi": 0}, "dpi must be a positive number"),
        ({"dpi": True}, "dpi must be a positive number"),
        ({"style": {"figure_size": [10, 8]}}, "only colormap and font_size"),
        ({"style": {"colormap": "not-a-colormap"}}, "is not a colormap"),
        ({"style": {"font_size": -1}}, "font_size must be a positive number"),
        ({"style": "dark"}, "only colormap and font_size"),
    ],
)
def test_invalid_plot_settings_are_rejected(tmp_path: Path, visualization, message) -> None:
    def mutate(config):
        config["visualization"] = {"output_dir": "output", **visualization}

    with pytest.raises(ValueError, match=message):
        _model_with(tmp_path, mutate)
