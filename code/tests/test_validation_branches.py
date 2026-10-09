"""Guard and fallback branches that decide what a malformed input does.

Each test pairs a rejected input with the nearest accepted one, so a guard that
fires too eagerly fails here just as a guard that never fires does.  Modules
covered: the discrete generative model, the homeostatic model and its
declarative components, the dispatcher factory, the base-model state checks,
``MatrixOps``/``MatrixInitializer``, the knowledge-network visualizer, and the
matrix plotters.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from cognitive.models.active_inference import generative_model as gm
from cognitive.models.active_inference.base import ActiveInferenceModel, ModelState
from cognitive.models.active_inference.dispatcher import (
    ActiveInferenceDispatcher,
    ActiveInferenceFactory,
    InferenceConfig,
)
from cognitive.models.active_inference.generative_model import DiscreteGenerativeModel
from cognitive.models.active_inference.homeostatic import (
    AdaptiveControl,
    HomeostaticFactory,
    HomeostaticInference,
    ObservationModel,
    StateSpace,
    TransitionModel,
)
from cognitive.models.matrices.matrix_ops import MatrixInitializer, MatrixLoader, MatrixOps
from cognitive.utils.create_node import NodeCreator
from cognitive.utils.matrix_utils import ensure_matrix_properties, kl_divergence, softmax
from cognitive.utils.visualization.network_viz import NetworkVisualizer
from cognitive.visualization.matrix_plots import MatrixPlotter, NetworkPlotter, StateSpacePlotter

# --- generative model -------------------------------------------------------

A2 = np.array([[0.9, 0.1], [0.1, 0.9]])
B2 = np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2)


def build(**overrides: Any) -> DiscreteGenerativeModel:
    arrays: dict[str, Any] = {
        "A": A2,
        "B": B2,
        "C": np.array([0.0, 1.0]),
        "D": np.array([0.5, 0.5]),
        "E": np.array([0.5, 0.5]),
    }
    arrays.update(overrides)
    return DiscreteGenerativeModel(**arrays)


def test_valid_model_exposes_its_dimensions_and_normalized_priors() -> None:
    model = build(D=np.array([1.0, 3.0]), E=np.array([2.0, 2.0]))
    assert (model.num_observations, model.num_states, model.num_actions) == (2, 2, 2)
    np.testing.assert_allclose(model.D, [0.25, 0.75])
    np.testing.assert_allclose(model.E, [0.5, 0.5])
    assert model.preferences == pytest.approx(np.exp([0.0, 1.0]) / np.exp([0.0, 1.0]).sum())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"A": np.ones(3) / 3}, "A must have shape"),
        ({"B": np.ones((2, 2))}, "B must have shape"),
        ({"B": np.ones((2, 3, 2)) / 2}, "equal current and next state"),
        ({"A": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])}, "A and B state dimensions"),
        ({"C": np.zeros(3)}, "one preference per observation"),
        ({"D": np.ones(3) / 3}, "one prior per state"),
        ({"E": np.ones(3) / 3}, "one prior per action"),
        ({"A": np.array([[1.5, 0.0], [-0.5, 1.0]])}, "A must be finite and non-negative"),
        ({"B": B2 * np.nan}, "B must be finite and non-negative"),
        ({"A": np.array([[0.5, 0.5], [0.4, 0.5]])}, "A must be column stochastic"),
        ({"B": B2 * 0.5}, "column must sum to one"),
        ({"C": np.array([0.0, np.inf])}, "C must contain finite log preferences"),
        ({"D": np.zeros(2)}, "D must contain positive mass"),
        ({"E": np.array([1.0, -1.0])}, "E must be a finite, non-negative"),
    ],
)
def test_model_construction_names_the_invalid_field(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build(**overrides)


def test_from_config_rejects_missing_and_unknown_fields() -> None:
    config = {"A": A2, "B": B2, "C": [0.0, 1.0], "D": [0.5, 0.5], "E": [0.5, 0.5]}
    assert DiscreteGenerativeModel.from_config(config).num_states == 2
    with pytest.raises(ValueError, match="Missing generative model fields"):
        DiscreteGenerativeModel.from_config({"A": A2})
    with pytest.raises(ValueError, match="Unknown generative model fields"):
        DiscreteGenerativeModel.from_config({**config, "F": [1.0]})


def test_prediction_methods_reject_beliefs_of_the_wrong_shape() -> None:
    model = build()
    wrong = np.ones(3) / 3
    with pytest.raises(ValueError, match="beliefs must have shape"):
        model.predict_states(wrong, 0)
    with pytest.raises(ValueError, match="beliefs must have shape"):
        model.predict_observations(wrong)
    with pytest.raises(ValueError, match="prior must have shape"):
        model.posterior(0, wrong)
    with pytest.raises(ValueError, match="Action out of range"):
        model.predict_states(model.D, 2)
    with pytest.raises(ValueError, match="observation must have shape"):
        model.posterior(np.ones(3) / 3, model.D)


def test_vector_observation_weights_the_likelihood_by_its_distribution() -> None:
    """A distribution over observations mixes the rows of ``A`` it names."""
    model = build()
    prior = np.array([0.5, 0.5])
    mixed = model.posterior(np.array([0.5, 0.5]), prior)
    assert mixed == pytest.approx(prior)
    assert model.posterior(np.array([1.0, 0.0]), prior) == pytest.approx(model.posterior(0, prior))


def test_probability_helpers_reject_bad_mass_and_fall_back_to_uniform() -> None:
    with pytest.raises(ValueError, match="finite, non-negative, non-empty"):
        gm._as_probability_vector(np.array([0.5, np.nan]), name="x")
    with pytest.raises(ValueError, match="positive mass"):
        gm._as_probability_vector(np.zeros(3), name="x")
    assert gm._normalize(np.zeros(4)) == pytest.approx(np.full(4, 0.25))
    assert gm._normalize(np.array([-1.0, 3.0])) == pytest.approx([0.0, 1.0])
    assert gm._entropy(np.zeros(4)) == pytest.approx(np.log(4.0))


# --- base model state -------------------------------------------------------


class ListConfigModel(ActiveInferenceModel):
    """Concrete model whose loader returns something other than a mapping."""

    def _load_config(self, config_path: Any) -> Any:
        return list(config_path)

    def _initialize_matrices(self) -> None:
        raise AssertionError("unreachable: configuration is rejected first")

    def _initialize_state(self) -> ModelState:
        raise AssertionError("unreachable: configuration is rejected first")

    def step(self, action: int | None = None) -> tuple[int, float]:
        raise NotImplementedError

    def visualize(self, plot_type: str, **kwargs: Any) -> Any:
        raise NotImplementedError


def test_model_configuration_must_be_a_mapping() -> None:
    with pytest.raises(ValueError, match="Model configuration must be a mapping"):
        ListConfigModel([1, 2])


@pytest.fixture
def homeostatic_model() -> HomeostaticInference:
    return HomeostaticFactory.create_basic(homeostatic_config())


def homeostatic_config() -> dict[str, Any]:
    return {
        "state_spaces": {
            name: {
                "dimensions": [2],
                "labels": {key: [f"{key}0", f"{key}1"]},
                "mappings": {"identity": np.eye(2).tolist()},
            }
            for name, key in (
                ("environment", "states"),
                ("observation", "observations"),
                ("action", "actions"),
            )
        },
        "observation_model": {"likelihood_matrix": A2.tolist()},
        "transition_model": {
            "transition_matrices": {"actions0": np.eye(2).tolist(), "actions1": [[0, 1], [1, 0]]},
            "temporal_horizon": 1,
        },
        "inference": {"learning_rate": 0.5, "precision_init": 1.0},
        "target_state": "states1",
    }


def test_saved_states_are_validated_field_by_field(
    homeostatic_model: HomeostaticInference, tmp_path: Path
) -> None:
    path = homeostatic_model.save_state(tmp_path / "state.yaml")
    good = yaml.safe_load(path.read_text())
    homeostatic_model.load_state(path)

    def reload(**changes: Any) -> None:
        payload = {**good, **changes}
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump(payload))
        homeostatic_model.load_state(bad)

    with pytest.raises(ValueError, match="beliefs must be a finite, non-empty vector"):
        reload(beliefs=[[0.5, 0.5]])
    with pytest.raises(ValueError, match="policies must be a normalized probability vector"):
        reload(policies=[0.9, 0.9])
    with pytest.raises(ValueError, match="free_energy must be finite"):
        reload(free_energy=float("nan"))
    with pytest.raises(ValueError, match="precision must be positive"):
        reload(precision=-1.0)


# --- homeostatic components -------------------------------------------------


def spaces() -> tuple[StateSpace, StateSpace, StateSpace]:
    def make(key: str, size: int) -> StateSpace:
        return StateSpace([size], {key: [f"{key}{i}" for i in range(size)]}, {"id": np.eye(size)})

    return make("states", 2), make("observations", 2), make("actions", 2)


def test_observation_model_rejects_each_malformed_input() -> None:
    states, observations, _ = spaces()
    with pytest.raises(ValueError, match="likelihood_matrix must have shape"):
        ObservationModel(states, observations, np.ones((3, 2)) / 3)
    with pytest.raises(ValueError, match="finite and non-negative"):
        ObservationModel(states, observations, np.array([[1.5, 0.0], [-0.5, 1.0]]))
    with pytest.raises(ValueError, match="precision must be finite and positive"):
        ObservationModel(states, observations, A2, precision=0.0)
    model = ObservationModel(states, observations, A2)
    with pytest.raises(ValueError, match="state must match"):
        model.compute_likelihood(np.array([1.0, 0.0]), np.ones(3) / 3)
    with pytest.raises(ValueError, match="observation must match"):
        model.compute_likelihood(np.ones(3), np.array([1.0, 0.0]))
    assert model.compute_likelihood(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == pytest.approx(
        0.9
    )


def test_transition_model_rejects_each_malformed_matrix() -> None:
    states, _, actions = spaces()
    identity = np.eye(2)
    for matrix, message in (
        (np.eye(3), "must have shape"),
        (np.array([[np.nan, 0.0], [1.0, 1.0]]), "finite and non-negative"),
        (np.array([[0.5, 0.5], [0.4, 0.5]]), "column stochastic"),
    ):
        with pytest.raises(ValueError, match=message):
            TransitionModel(states, actions, {"actions0": identity, "actions1": matrix}, 1)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.update(state_spaces={"environment": {}}), "must define environment"),
        (lambda c: c["state_spaces"].update(action=[]), "state_spaces.action must be a mapping"),
        (
            lambda c: c["state_spaces"]["action"].update(extra=1),
            "Unknown state_spaces.action fields",
        ),
        (lambda c: c.update(observation_model=[]), "observation_model must be a mapping"),
        (lambda c: c.update(transition_model=[]), "transition_model must be a mapping"),
        (
            lambda c: c["transition_model"].update(extra=1),
            "Unknown transition_model fields",
        ),
        (
            lambda c: c["transition_model"].update(transition_matrices=[]),
            "transition_matrices must be a mapping",
        ),
        (lambda c: c.update(inference=[]), "inference must be a mapping"),
    ],
)
def test_homeostatic_configuration_sections_are_type_checked(mutate: Any, message: str) -> None:
    config = copy.deepcopy(homeostatic_config())
    mutate(config)
    with pytest.raises(ValueError, match=message):
        HomeostaticFactory.create_basic(config)


def test_homeostatic_configuration_file_must_hold_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a mapping\n")
    with pytest.raises(ValueError, match="Homeostatic configuration must be a mapping"):
        HomeostaticInference(path, AdaptiveControl())


def test_factory_reads_the_target_from_a_configuration_file(tmp_path: Path) -> None:
    config = homeostatic_config()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    model = HomeostaticFactory.create_basic(path)
    assert model.control_mode.target_state == "states1"
    assert model._target_vector() == pytest.approx([0.0, 1.0])


def test_space_construction_reports_missing_and_inconsistent_spaces() -> None:
    config = homeostatic_config()
    with pytest.raises(ValueError, match="Missing state space: observation"):
        HomeostaticInference._space_from_config({"environment": {}}, "observation")
    broken = copy.deepcopy(config["state_spaces"])
    broken["environment"]["labels"] = {"states": ["only-one"]}
    with pytest.raises(ValueError, match="Invalid state space: environment"):
        HomeostaticInference._space_from_config(broken, "environment")


def test_homeostatic_vector_helpers_validate_and_normalize(
    homeostatic_model: HomeostaticInference,
) -> None:
    normalize = HomeostaticInference._normalize
    assert normalize([1.0, 3.0], 2) == pytest.approx([0.25, 0.75])
    assert normalize(np.zeros(3)) == pytest.approx(np.full(3, 1 / 3))
    assert normalize(np.zeros(3), 3) == pytest.approx(np.full(3, 1 / 3))
    with pytest.raises(ValueError, match="Expected vector of size 3"):
        normalize([1.0, 2.0], 3)
    with pytest.raises(ValueError, match="finite and non-negative"):
        normalize([1.0, -2.0], 2)
    with pytest.raises(ValueError, match="Observation out of range"):
        homeostatic_model._observation_vector(7, 2)
    assert homeostatic_model._observation_vector([1.0, 3.0], 2) == pytest.approx([0.25, 0.75])


def test_target_defaults_to_the_uniform_distribution_without_a_configured_target() -> None:
    config = homeostatic_config()
    del config["target_state"]
    model = HomeostaticInference(config, AdaptiveControl())
    assert model._target_vector() == pytest.approx([0.5, 0.5])


# --- dispatcher factory -----------------------------------------------------


def dispatcher_document() -> dict[str, Any]:
    return {
        "method": "variational",
        "policy_type": "discrete",
        "temporal_horizon": 1,
        "learning_rate": 0.5,
        "precision_init": 1.0,
        "generative_model": {
            "A": A2.tolist(),
            "B": B2.tolist(),
            "C": [0.0, 1.0],
            "D": [0.5, 0.5],
            "E": [0.5, 0.5],
        },
    }


def test_factory_builds_a_dispatcher_from_a_document(tmp_path: Path) -> None:
    path = tmp_path / "dispatcher.yaml"
    path.write_text(yaml.safe_dump(dispatcher_document()))
    dispatcher = ActiveInferenceFactory.create_from_yaml(path)
    assert dispatcher.model.num_states == 2
    assert dispatcher.config.temperature == 1.0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- a\n- b\n", "Dispatcher configuration must be a mapping"),
        ("method: variational\n", "must contain a generative_model mapping"),
    ],
)
def test_factory_rejects_documents_without_a_model(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        ActiveInferenceFactory.create_from_yaml(path)


def test_factory_names_missing_and_unknown_configuration_fields(tmp_path: Path) -> None:
    document = dispatcher_document()
    del document["learning_rate"]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=r"Missing dispatcher configuration fields: \['learning"):
        ActiveInferenceFactory.create_from_yaml(path)

    document = dispatcher_document()
    document["surprise"] = 1
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="Unknown dispatcher configuration fields"):
        ActiveInferenceFactory.create_from_yaml(path)


def test_dispatcher_normalization_checks_size_and_falls_back_to_uniform() -> None:
    normalize = ActiveInferenceDispatcher._normalize
    assert normalize(np.array([1.0, 3.0]), 2) == pytest.approx([0.25, 0.75])
    assert normalize(np.zeros(3), 3) == pytest.approx(np.full(3, 1 / 3))
    with pytest.raises(ValueError, match="Expected 3 finite values"):
        normalize(np.array([1.0, 2.0]), 3)
    with pytest.raises(ValueError, match="Expected 2 finite values"):
        normalize(np.array([1.0, np.nan]), 2)
    config = InferenceConfig("variational", "discrete", 1, 0.5, 1.0)
    assert config.policy_aggregation.value == "min"


# --- matrix_utils / matrix_ops ----------------------------------------------


def test_probability_helpers_reject_invalid_distributions() -> None:
    with pytest.raises(ValueError, match="p must be finite and non-negative"):
        kl_divergence(np.array([0.5, -0.5]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="q must contain positive mass"):
        kl_divergence(np.array([0.5, 0.5]), np.zeros(2))
    with pytest.raises(ValueError, match="same shape"):
        kl_divergence(np.ones(2), np.ones(3))
    assert kl_divergence(np.array([1.0, 3.0]), np.array([1.0, 3.0])) == pytest.approx(
        0.0, abs=1e-12
    )
    with pytest.raises(ValueError, match="x must be finite"):
        softmax(np.array([0.0, np.inf]))
    assert softmax(np.array([1000.0, 1000.0])) == pytest.approx([0.5, 0.5])


def test_constraints_that_are_not_requested_leave_the_matrix_unchanged() -> None:
    matrix = np.array([[2.0, -1.0], [3.0, 5.0]])
    assert ensure_matrix_properties(matrix) == pytest.approx(matrix)
    assert ensure_matrix_properties(matrix, []) == pytest.approx(matrix)
    assert ensure_matrix_properties(matrix, "non_negative") == pytest.approx(
        np.array([[2.0, 0.0], [3.0, 5.0]])
    )
    row_normalized = ensure_matrix_properties(np.array([[1.0, 3.0]]), "row_stochastic")
    np.testing.assert_allclose(row_normalized, [[0.25, 0.75]])


def test_matrix_loader_requires_a_mapping_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "spec.md"
    path.write_text("---\n- just\n- a list\n---\n")
    with pytest.raises(ValueError, match="frontmatter must be a mapping"):
        MatrixLoader.load_spec(path)
    path.write_text("---\ntype: spec\n")
    with pytest.raises(ValueError, match="terminator"):
        MatrixLoader.load_spec(path)
    path.write_text("no frontmatter at all")
    assert MatrixLoader.load_spec(path) == {}


def test_matrix_initializers_validate_shape_and_strength() -> None:
    with pytest.raises(ValueError, match="positive"):
        MatrixInitializer.random_stochastic((2, 0))
    with pytest.raises(ValueError, match="positive"):
        MatrixInitializer.uniform((0, 2))
    with pytest.raises(ValueError, match="square, positive"):
        MatrixInitializer.identity_based((2, 3))
    with pytest.raises(ValueError, match="strength"):
        MatrixInitializer.identity_based((2, 2), strength=1.5)
    seeded = MatrixInitializer.random_stochastic((3, 4), np.random.default_rng(5))
    assert seeded == pytest.approx(
        MatrixInitializer.random_stochastic((3, 4), np.random.default_rng(5))
    )
    assert seeded.sum(axis=0) == pytest.approx(np.ones(4))
    np.testing.assert_allclose(MatrixInitializer.identity_based((1, 1)), [[1.0]])
    assert MatrixInitializer.identity_based((3, 3), 0.7).sum(axis=0) == pytest.approx(np.ones(3))
    assert MatrixOps.compute_kl_divergence(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == (
        pytest.approx(np.log(2.0), abs=1e-9)
    )


# --- node creation ----------------------------------------------------------


def test_node_creator_rejects_a_non_mapping_configuration_and_unsafe_names(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("- not a mapping\n")
    with pytest.raises(ValueError, match="Configuration must be a YAML mapping"):
        NodeCreator(config)

    (tmp_path / "templates").mkdir()
    absolute_kb = tmp_path / "vault"
    config.write_text(
        yaml.safe_dump({"paths": {"templates": "templates", "knowledge_base": str(absolute_kb)}})
    )
    creator = NodeCreator(config)
    assert creator.knowledge_base_path == absolute_kb.resolve()
    with pytest.raises(ValueError, match="filename-safe"):
        creator._safe_name("...")


# --- knowledge network ------------------------------------------------------


def write_network(root: Path) -> Path:
    vault = root / "vault"
    (vault / "agents").mkdir(parents=True)
    (vault / "agents" / "alpha.md").write_text(
        "---\ntitle: Alpha\n---\n"
        "[[Beta|alias]] [[beta#section]] [[alpha]] [[|empty]] [[#only-anchor]] [[Missing]]\n"
    )
    (vault / "agents" / "beta.md").write_text("---\n- a\n- list\n---\n[[alpha]]\n")
    (vault / "agents" / "gamma.md").write_text("---\nunterminated: true\n[[beta]]\n")
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump({"paths": {"knowledge_base": "vault"}}))
    return config


def test_network_resolves_aliases_and_anchors_without_self_loops(tmp_path: Path) -> None:
    graph = NetworkVisualizer(write_network(tmp_path)).build_network()
    assert sorted(graph.nodes) == ["agents/alpha", "agents/beta", "agents/gamma"]
    assert sorted(tuple(sorted(edge)) for edge in graph.edges) == [
        ("agents/alpha", "agents/beta"),
        ("agents/beta", "agents/gamma"),
    ]
    assert graph.nodes["agents/alpha"]["title"] == "Alpha"
    assert {data["type"] for _, data in graph.nodes(data=True)} == {"agent"}
    assert "links" not in graph.nodes["agents/alpha"]


def test_network_renders_html_and_handles_degenerate_graphs(tmp_path: Path) -> None:
    visualizer = NetworkVisualizer(write_network(tmp_path))
    output = tmp_path / "out" / "network.html"
    figure = visualizer.visualize(output)
    assert output.is_file()
    assert len(figure.data) == 2

    empty = NetworkVisualizer(tmp_path / "config.yaml")
    empty.knowledge_base_path = tmp_path / "does-not-exist"
    assert len(empty.build_network()) == 0
    assert empty._get_node_positions(empty.build_network()) == {}

    single = tmp_path / "single"
    single.mkdir()
    (single / "only.md").write_text("no links\n")
    empty.knowledge_base_path = single
    graph = empty.build_network()
    assert empty._get_node_positions(graph) == {"only": (0.0, 0.0)}
    with pytest.raises(ValueError, match="width and height"):
        empty.visualize(width=0)

    absolute = tmp_path / "absolute.yaml"
    absolute.write_text(yaml.safe_dump({"paths": {"knowledge_base": str(single)}}))
    assert NetworkVisualizer(absolute).knowledge_base_path == single.resolve()


# --- plotting ---------------------------------------------------------------


def test_matrix_plotter_draws_onto_a_supplied_axes_and_saves_by_name(tmp_path: Path) -> None:
    plotter = MatrixPlotter(str(tmp_path), {"file_format": "png"})
    figure, axes = plt.subplots()
    returned = plotter.plot_heatmap(
        np.eye(3), ax=axes, title="T", xlabel="X", ylabel="Y", cmap="Blues", save_name="heat"
    )
    assert returned is figure
    assert (tmp_path / "heat.png").is_file()
    assert axes.get_title() == "T"
    assert (axes.get_xlabel(), axes.get_ylabel()) == ("X", "Y")

    figure, axes = plt.subplots()
    plotter.plot_bar(np.array([0.2, 0.8]), ax=axes, title="B", save_name="bar")
    assert (tmp_path / "bar.png").is_file()
    assert [patch.get_height() for patch in axes.patches] == pytest.approx([0.2, 0.8])

    single = plotter.plot_multi_heatmap(np.ones((1, 2, 2)), "T", "x", "y", ["only"], save_name="m")
    assert (tmp_path / "m.png").is_file()
    assert single.axes[0].get_title() == "T - only"
    plt.close("all")


def test_free_energy_landscape_marks_points_outside_the_simplex(tmp_path: Path) -> None:
    plotter = MatrixPlotter(str(tmp_path), {})
    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")
    returned = plotter.plot_free_energy_landscape(
        lambda beliefs: float(-np.sum(beliefs * np.log(beliefs + 1e-12))), num_points=6, ax=axes
    )
    assert returned is axes
    assert axes.get_zlabel() == "Expected Free Energy"
    plt.close("all")


def test_state_space_plotters_without_a_directory_do_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    states = StateSpacePlotter(None)
    assert states.save_dir is None
    figures = [
        states.plot_belief_evolution(np.eye(2), "t", ["a", "b"], save_name="ignored"),
        states.plot_free_energy_landscape(np.ones((3, 3)), "t", save_name="ignored"),
        states.plot_policy_evaluation(np.array([1.0, 2.0]), ["p0", "p1"], "t", "ignored"),
        NetworkPlotter().plot_belief_network(np.eye(2), ["a", "b"], "t"),
    ]
    assert all(figure.axes for figure in figures)
    assert not list(tmp_path.glob("*.png"))
    plt.close("all")


# --- continuous visualizer and Simple_POMDP matrices -------------------------


def test_continuous_visualizer_validates_histories_before_drawing(tmp_path: Path) -> None:
    from Things.Continuous_Generic.visualization import ContinuousVisualizer

    visualizer = ContinuousVisualizer(tmp_path)
    means = np.zeros((4, 2, 3))
    with pytest.raises(ValueError, match="history must contain belief_means"):
        visualizer.save_animation({}, tmp_path / "a.gif")
    with pytest.raises(ValueError, match=r"shape \(frames, states, orders\)"):
        visualizer.save_animation({"belief_means": np.zeros((4, 2))}, tmp_path / "a.gif")
    with pytest.raises(ValueError, match="belief_means must be finite"):
        visualizer.save_animation({"belief_means": means * np.nan}, tmp_path / "a.gif")
    with pytest.raises(ValueError, match="one finite value per frame"):
        visualizer.save_animation({"belief_means": means, "time": [0.0, 1.0]}, tmp_path / "a.gif")
    with pytest.raises(ValueError, match="fps must be a positive integer"):
        visualizer.save_animation({"belief_means": means}, tmp_path / "a.gif", fps=0)
    with pytest.raises(ValueError, match="share a 3-D shape"):
        visualizer.plot_belief_evolution([means[0]], [means], tmp_path / "b.png")
    with pytest.raises(ValueError, match="match the history length"):
        visualizer.create_summary_plot(
            {"belief_means": means, "actions": [0.0] * 3, "free_energy": [0.0] * 4},
            tmp_path / "s.png",
        )

    gif = visualizer.save_animation({"belief_means": means}, tmp_path / "ok.gif", fps=5)
    assert gif.is_file()
    plt.close("all")


def test_summary_plot_draws_a_single_state_history_without_a_phase_portrait(
    tmp_path: Path,
) -> None:
    from Things.Continuous_Generic.visualization import ContinuousVisualizer

    history = {
        "belief_means": np.zeros((3, 1, 2)),
        "actions": [0.0, 0.1, 0.2],
        "free_energy": [1.0, 0.5, 0.25],
    }
    figure = ContinuousVisualizer(tmp_path).create_summary_plot(history, tmp_path / "s.png")
    phase_axes = figure.axes[1]
    assert phase_axes.get_title() == "Phase Space"
    assert not phase_axes.lines
    assert (tmp_path / "s.png").is_file()


def test_simple_pomdp_rejects_non_finite_preferences_and_mismatched_shapes(tmp_path: Path) -> None:
    from Things.Simple_POMDP import SimplePOMDP

    def config() -> dict[str, Any]:
        return {
            "model": {"name": "m", "version": "1"},
            "state_space": {"num_states": 2},
            "observation_space": {"num_observations": 2},
            "action_space": {"num_actions": 2},
            "matrices": {
                "A_matrix": {"shape": [2, 2], "initialization": "identity_based"},
                "B_matrix": {"shape": [2, 2, 2], "initialization": "identity_based"},
                "C_matrix": {
                    "shape": [2],
                    "initialization": "log_preferences",
                    "initialization_params": {"preferences": [0.0, 1.0]},
                },
                "D_matrix": {"shape": [2], "initialization": "uniform"},
                "E_matrix": {"shape": [2], "initialization": "uniform"},
            },
            "inference": {},
            "visualization": {"output_dir": str(tmp_path / "plots")},
        }

    assert SimplePOMDP(config()).C.shape == (2,)

    bad_c = config()
    bad_c["matrices"]["C_matrix"]["initialization_params"]["preferences"] = [0.0, float("inf")]
    with pytest.raises(ValueError, match="finite preferences"):
        SimplePOMDP(bad_c)

    mismatched = config()
    mismatched["observation_space"]["num_observations"] = 3
    with pytest.raises(ValueError, match="A matrix shape must match"):
        SimplePOMDP(mismatched)
