"""Regression tests for input boundaries and numerical edge cases.

Each test here failed against the code as it stood before the corresponding fix:
an index that ``int()`` silently truncated, a NaN column that turned into a
uniform distribution, a posterior discarded because its evidence was small, an
entropy that dipped below zero, and plots that failed or varied for reasons
unrelated to their data.  Every assertion compares against a value derived
independently of the implementation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from cognitive.models.active_inference.base import ActiveInferenceModel
from cognitive.models.active_inference.dispatcher import InferenceConfig
from cognitive.models.active_inference.generative_model import DiscreteGenerativeModel
from cognitive.models.active_inference.homeostatic import HomeostaticFactory
from cognitive.utils.matrix_utils import (
    as_integer,
    compute_entropy,
    ensure_matrix_properties,
    expected_free_energy,
)
from cognitive.visualization.matrix_plots import MatrixPlotter, NetworkPlotter, StateSpacePlotter
from Things.Continuous_Generic.visualization import ContinuousVisualizer
from Things.Simple_POMDP import SimplePOMDP


def two_state_model(likelihood: np.ndarray | None = None) -> DiscreteGenerativeModel:
    return DiscreteGenerativeModel(
        A=np.array([[0.9, 0.1], [0.1, 0.9]]) if likelihood is None else likelihood,
        B=np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2),
        C=np.array([0.0, 1.0]),
        D=np.array([0.5, 0.5]),
        E=np.array([0.5, 0.5]),
    )


# --- as_integer -------------------------------------------------------------


@pytest.mark.parametrize("value", [2, np.int64(2), 2.0, np.float64(2.0), True])
def test_as_integer_accepts_integral_values(value: object) -> None:
    assert as_integer(value, "index") == int(value)


@pytest.mark.parametrize("value", [1.9, -0.5, float("nan"), float("inf"), "a", None, [1]])
def test_as_integer_rejects_everything_else(value: object) -> None:
    with pytest.raises(ValueError, match="index must be an integer"):
        as_integer(value, "index")


# --- matrix_utils -----------------------------------------------------------


def test_entropy_of_a_deterministic_distribution_is_exactly_zero() -> None:
    """A log floor applied to every entry made ``H([1, 0])`` about -1e-12."""
    assert compute_entropy(np.array([1.0, 0.0])) == 0.0
    assert np.all(compute_entropy(np.eye(3), axis=0) == 0.0)
    assert np.all(compute_entropy(np.eye(3), axis=1) == 0.0)


def test_entropy_matches_the_closed_form_away_from_the_boundary() -> None:
    probabilities = np.array([0.5, 0.25, 0.25])
    assert compute_entropy(probabilities) == pytest.approx(1.5 * np.log(2.0), abs=1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[np.nan, 0.5], [0.5, 0.5]]),
        np.array([[np.inf, 0.5], [0.5, 0.5]]),
        np.empty((0, 2)),
    ],
)
def test_matrix_constraints_reject_non_finite_and_empty_input(bad: np.ndarray) -> None:
    """A NaN column used to fail ``sum > EPS`` and be replaced by a uniform one."""
    with pytest.raises(ValueError, match="non-empty and finite"):
        ensure_matrix_properties(bad, "column_stochastic")


@pytest.mark.parametrize("constraint", ["column_stochastic", "row_stochastic"])
def test_stochastic_constraints_require_non_negative_entries(constraint: str) -> None:
    """Normalizing a negative entry returned a 'stochastic' matrix with negatives."""
    matrix = np.array([[-1.0, 0.5], [3.0, 0.5]])
    with pytest.raises(ValueError, match="non-negative"):
        ensure_matrix_properties(matrix, constraint)
    clipped = ensure_matrix_properties(matrix, ["non_negative", constraint])
    assert np.all(clipped >= 0.0)
    axis = 0 if constraint == "column_stochastic" else 1
    assert clipped.sum(axis=axis) == pytest.approx(1.0)


def test_all_zero_columns_still_fall_back_to_uniform() -> None:
    """Zero mass is a legitimate degenerate input, unlike NaN or negative mass."""
    result = ensure_matrix_properties(np.array([[0.0, 1.0], [0.0, 3.0]]), "column_stochastic")
    assert result == pytest.approx(np.array([[0.5, 0.25], [0.5, 0.75]]))


@pytest.mark.parametrize("action", [1.9, 0.5, float("nan")])
def test_expected_free_energy_helper_rejects_fractional_actions(action: float) -> None:
    """``int(1.9)`` is 1, so the helper used to evaluate a different action."""
    model = two_state_model()
    with pytest.raises(ValueError, match="action must be an integer"):
        expected_free_energy(model.A, model.B, model.C, model.D, action)


def test_expected_free_energy_helper_accepts_integral_actions_of_any_type() -> None:
    model = two_state_model()
    reference = expected_free_energy(model.A, model.B, model.C, model.D, 1)
    assert expected_free_energy(model.A, model.B, model.C, model.D, np.int64(1)) == reference
    assert expected_free_energy(model.A, model.B, model.C, model.D, 1.0) == reference
    with pytest.raises(ValueError, match="action must be valid"):
        expected_free_energy(model.A, model.B, model.C, model.D, 2)


def test_helper_and_model_agree_exactly_when_the_likelihood_is_deterministic() -> None:
    """With a noiseless likelihood the ambiguity is exactly zero in both."""
    model = two_state_model(np.eye(2))
    for action in range(model.num_actions):
        total, _risk, ambiguity, _epistemic = model.expected_free_energy(model.D, action)
        assert ambiguity == 0.0
        assert expected_free_energy(model.A, model.B, model.C, model.D, action) == pytest.approx(
            total, abs=1e-15
        )


# --- generative model -------------------------------------------------------


@pytest.mark.parametrize("observation", [1.7, 0.2, float("nan")])
def test_posterior_rejects_fractional_observation_indices(observation: float) -> None:
    with pytest.raises(ValueError, match="Observation must be an integer"):
        two_state_model().posterior(observation, np.array([0.5, 0.5]))


def test_posterior_treats_a_zero_dimensional_array_as_an_index() -> None:
    """A 0-d array is a scalar index, not a one-element observation vector."""
    model = two_state_model()
    prior = np.array([0.5, 0.5])
    expected = model.posterior(1, prior)
    assert model.posterior(np.array(1), prior) == pytest.approx(expected)
    assert model.posterior(np.float64(1.0), prior) == pytest.approx(expected)
    assert expected == pytest.approx([0.1, 0.9])
    with pytest.raises(ValueError, match="Observation out of range"):
        model.posterior(np.array(5), prior)


def test_posterior_is_exact_for_small_positive_evidence() -> None:
    """Evidence below 1e-12 was treated as impossible and the prior returned.

    Bayes' rule is scale-invariant, so likelihoods of 1e-13 and 1e-14 give the
    same posterior as 10 and 1: ``(10/11, 1/11)``.
    """
    small = 1e-13
    likelihood = np.array([[small, small / 10.0], [1.0 - small, 1.0 - small / 10.0]])
    model = two_state_model(likelihood)
    posterior = model.posterior(0, np.array([0.5, 0.5]))
    assert posterior == pytest.approx([10.0 / 11.0, 1.0 / 11.0], abs=1e-9)
    skewed = model.posterior(0, np.array([0.2, 0.8]))
    unnormalized = np.array([0.2 * small, 0.8 * small / 10.0])
    assert skewed == pytest.approx(unnormalized / unnormalized.sum(), abs=1e-9)


def test_posterior_returns_the_prior_when_the_observation_is_impossible() -> None:
    """Zero evidence has no posterior; the documented fallback is the prior."""
    model = two_state_model(np.eye(2))
    prior = np.array([0.0, 1.0])
    assert model.posterior(0, prior) == pytest.approx(prior)


@pytest.mark.parametrize("discount", [0.0, -0.5, 1.5, float("nan"), float("inf")])
def test_policy_evaluation_validates_the_discount_factor(discount: float) -> None:
    with pytest.raises(ValueError, match="discount_factor must be in"):
        two_state_model().evaluate_policy(np.array([0.5, 0.5]), (0, 1), discount)


def test_policy_evaluation_rejects_an_empty_policy() -> None:
    """An empty policy used to evaluate to a free energy of exactly zero."""
    with pytest.raises(ValueError, match="at least one action"):
        two_state_model().evaluate_policy(np.array([0.5, 0.5]), (), 0.9)


# --- base and homeostatic ---------------------------------------------------


def test_base_observation_vector_rejects_fractional_and_accepts_zero_dimensional() -> None:
    with pytest.raises(ValueError, match="Observation must be an integer"):
        ActiveInferenceModel._observation_vector(1.5, 3)
    assert ActiveInferenceModel._observation_vector(np.array(2), 3) == pytest.approx([0, 0, 1])


def homeostatic_config() -> dict[str, object]:
    return {
        "state_spaces": {
            "environment": {
                "dimensions": [2],
                "labels": {"states": ["low", "high"]},
                "mappings": {"identity": np.eye(2).tolist()},
            },
            "observation": {
                "dimensions": [2],
                "labels": {"observations": ["dim", "bright"]},
                "mappings": {"identity": np.eye(2).tolist()},
            },
            "action": {
                "dimensions": [2],
                "labels": {"actions": ["hold", "move"]},
                "mappings": {"identity": np.eye(2).tolist()},
            },
        },
        "observation_model": {"likelihood_matrix": [[0.8, 0.2], [0.2, 0.8]]},
        "transition_model": {
            "transition_matrices": {"hold": np.eye(2).tolist(), "move": [[0, 1], [1, 0]]},
            "temporal_horizon": 1,
        },
        "inference": {
            "method": "variational",
            "policy_type": "discrete",
            "temporal_horizon": 1,
            "learning_rate": 0.5,
            "precision_init": 1.0,
        },
        "target_state": "high",
    }


def test_homeostatic_step_rejects_fractional_actions() -> None:
    """``step(1.9)`` used to run action 1 and report success."""
    model = HomeostaticFactory.create_basic(homeostatic_config())
    before = model.state.beliefs.copy()
    with pytest.raises(ValueError, match="Action must be an integer"):
        model.step(1.9)
    assert model.state.beliefs == pytest.approx(before)
    observation, _free_energy = model.step(1.0)
    assert observation in (0, 1)


def test_homeostatic_observation_vector_rejects_fractional_indices() -> None:
    model = HomeostaticFactory.create_basic(homeostatic_config())
    with pytest.raises(ValueError, match="Observation must be an integer"):
        model.update_beliefs(0.5)
    assert model._observation_vector(np.array(1), 2) == pytest.approx([0.0, 1.0])


def test_homeostatic_exploration_default_matches_the_inference_config_default() -> None:
    """1.1.0 moved the default to 0.0; the homeostatic model still applied 0.5."""
    model = HomeostaticFactory.create_basic(homeostatic_config())
    default = InferenceConfig("variational", "discrete", 1, 0.5, 1.0).exploration_weight
    assert default == 0.0
    assert model.dispatcher.config.exploration_weight == default

    explicit = homeostatic_config()
    explicit["inference"]["exploration_weight"] = 0.25
    configured = HomeostaticFactory.create_basic(explicit)
    assert configured.dispatcher.config.exploration_weight == 0.25


# --- Simple_POMDP -----------------------------------------------------------


def pomdp_config(tmp_path: Path, likelihood: list[list[float]]) -> dict[str, object]:
    return {
        "model": {"name": "boundary", "version": "1"},
        "state_space": {"num_states": 2, "state_labels": ["a", "b"]},
        "observation_space": {"num_observations": 2, "observation_labels": ["x", "y"]},
        "action_space": {"num_actions": 2, "action_labels": ["stay", "switch"]},
        "matrices": {
            "A_matrix": {
                "shape": [2, 2],
                "initialization": "log_preferences",
                "initialization_params": {"preferences": likelihood},
            },
            "B_matrix": {
                "shape": [2, 2, 2],
                "initialization": "identity_based",
                "initialization_params": {"strength": 0.8},
            },
            "C_matrix": {
                "shape": [2],
                "initialization": "log_preferences",
                "initialization_params": {"preferences": [0.0, 1.0]},
            },
            "D_matrix": {"shape": [2], "initialization": "uniform"},
            "E_matrix": {"shape": [2], "initialization": "uniform"},
        },
        "inference": {"learning_rate": 1.0},
        "visualization": {"output_dir": str(tmp_path / "plots")},
        "seed": 3,
    }


def test_simple_pomdp_step_rejects_fractional_actions(tmp_path: Path) -> None:
    model = SimplePOMDP(pomdp_config(tmp_path, [[0.9, 0.1], [0.1, 0.9]]))
    with pytest.raises(ValueError, match="Action must be an integer"):
        model.step(0.7)
    assert model.state.time_step == 0
    model.step(1.0)
    assert model.state.time_step == 1


def test_simple_pomdp_belief_update_is_exact_for_small_likelihoods(tmp_path: Path) -> None:
    """A posterior mass under 1e-12 was replaced by a uniform belief."""
    small = 1e-13
    model = SimplePOMDP(
        pomdp_config(tmp_path, [[small, small / 10.0], [1.0 - small, 1.0 - small / 10.0]])
    )
    prior = model.state.beliefs.copy()
    predicted = model.B[:, :, 0] @ prior
    unnormalized = predicted * model.A[0, :]
    model._update_beliefs(0, 0)
    assert model.state.beliefs == pytest.approx(unnormalized / unnormalized.sum(), abs=1e-9)
    assert not np.allclose(model.state.beliefs, [0.5, 0.5])


def test_simple_pomdp_rejects_fractional_initial_state_and_saved_indices(tmp_path: Path) -> None:
    """``initial_state`` and the persisted indices were cast with ``int`` and truncated."""
    config = pomdp_config(tmp_path, [[0.9, 0.1], [0.1, 0.9]])
    config["state_space"]["initial_state"] = 1.9  # type: ignore[index]
    with pytest.raises(ValueError, match="initial_state must be an integer"):
        SimplePOMDP(config)

    model = SimplePOMDP(pomdp_config(tmp_path, [[0.9, 0.1], [0.1, 0.9]]))
    saved = model.save_state(tmp_path / "state.yaml")
    text = saved.read_text(encoding="utf-8")
    assert "current_state: 0" in text and "time_step: 0" in text
    for field, value in (("current_state", "0.9"), ("time_step", "1.9")):
        corrupted = tmp_path / f"{field}.yaml"
        corrupted.write_text(
            text.replace(f"{field}: 0", f"{field}: {value}"),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=f"{field} must be an integer"):
            model.load_state(corrupted)


def test_state_space_plotter_names_missing_state_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="state_labels must name each of the 3 states"):
        StateSpacePlotter(tmp_path).plot_belief_evolution(np.eye(3), "beliefs", ["a"])


def test_evaluate_policy_accepts_iterables_and_rejects_non_numeric_discounts() -> None:
    model = DiscreteGenerativeModel(
        A=np.array([[0.9, 0.1], [0.1, 0.9]]),
        B=np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2),
        C=np.array([0.0, 1.0]),
        D=np.array([0.5, 0.5]),
        E=np.array([0.5, 0.5]),
    )
    from_tuple = model.evaluate_policy(model.D, (1, 0), 0.9)
    assert model.evaluate_policy(model.D, iter([1, 0]), 0.9) == pytest.approx(from_tuple)
    for bad in ("0.9", None):
        with pytest.raises(ValueError, match="discount_factor"):
            model.evaluate_policy(model.D, (1, 0), discount_factor=bad)  # type: ignore[arg-type]


# --- plotting ---------------------------------------------------------------


def test_free_energy_landscape_accepts_a_non_square_array(tmp_path: Path) -> None:
    """The default ``meshgrid`` indexing transposed X and Y, so Z never matched."""
    values = np.arange(15.0).reshape(3, 5)
    figure = StateSpacePlotter(tmp_path).plot_free_energy_landscape(values, "land", "land")
    assert (tmp_path / "land.png").is_file()
    assert len(figure.axes[0].collections) == 1
    plt.close(figure)


def _node_positions(figure: plt.Figure) -> np.ndarray:
    return np.asarray(figure.axes[0].collections[0].get_offsets())


def test_message_network_layout_is_reproducible(tmp_path: Path) -> None:
    connections = {"a": ["b", "c"], "b": ["c", "d"], "d": ["a"]}
    layouts = []
    for seed in (4, 4, 5):
        figure = NetworkPlotter(tmp_path, seed=seed).plot_message_network_dynamics(
            connections, [{"source": "a", "target": "b"}]
        )
        layouts.append(_node_positions(figure))
        plt.close(figure)
    assert np.array_equal(layouts[0], layouts[1])
    assert not np.allclose(layouts[0], layouts[2])


def test_plotters_name_the_missing_labels(tmp_path: Path) -> None:
    plotter = MatrixPlotter(str(tmp_path), {})
    with pytest.raises(ValueError, match="slice_names must name each of the 3 slices"):
        plotter.plot_multi_heatmap(np.ones((3, 2, 2)), "t", "x", "y", ["a", "b"])
    with pytest.raises(ValueError, match="state_labels must name each of the 2 states"):
        plotter.plot_belief_evolution(np.ones((3, 2)), state_labels=["only"])
    axes = plotter.plot_belief_evolution(np.ones((3, 2)), state_labels=["a", "b"])
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ["a", "b"]
    plt.close("all")


@pytest.mark.parametrize("missing", ["actions", "free_energy"])
def test_summary_plot_names_the_missing_history_key(tmp_path: Path, missing: str) -> None:
    """A missing key raised ``IndexError: tuple index out of range``."""
    history: dict[str, object] = {
        "belief_means": np.zeros((3, 2, 2)),
        "actions": [0.0, 0.1, 0.2],
        "free_energy": [1.0, 0.5, 0.25],
    }
    del history[missing]
    with pytest.raises(ValueError, match=f"history must contain {missing}"):
        ContinuousVisualizer(tmp_path).create_summary_plot(history, tmp_path / "summary.png")


# --- link checkers ----------------------------------------------------------


def test_a_comment_inside_a_code_fence_is_not_a_heading_anchor(tmp_path: Path) -> None:
    """``# Install deps`` in a bash block was accepted as the anchor ``#install-deps``."""
    from scripts import check_markdown_links, verify_links

    (tmp_path / "target.md").write_text(
        "# Real heading\n\n```bash\n# Install deps\npip install x\n```\n\n"
        "~~~\n# Tilde comment\n~~~\n"
    )
    (tmp_path / "source.md").write_text(
        "[[target#install-deps]] [[target#real-heading]] [[target#tilde-comment]]\n"
        "[bad](target.md#install-deps) [good](target.md#real-heading)\n"
    )
    wiki = verify_links.verify_link_report(tmp_path)
    assert sorted(w["fragment"] for w in wiki.anchor_warnings) == [
        "install-deps",
        "tilde-comment",
    ]
    standard = check_markdown_links.verify_markdown_link_report(tmp_path)
    assert [w["target"] for w in standard.anchor_warnings] == ["target.md#install-deps"]
