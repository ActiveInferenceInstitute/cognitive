"""Configuration and persistence guards for ``SimplePOMDP`` and the link gates.

Every assertion drives a rejection path. A validator whose failure branches are
never executed is a validator nobody has checked, so these tests pass malformed
input in deliberately and pin the message each guard produces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scripts.check_markdown_links import verify_markdown_link_report
from scripts.verify_links import verify_link_report
from Things.Simple_POMDP import SimplePOMDP


def pomdp_config(**overrides: object) -> dict[str, object]:
    """A minimal valid two-state configuration."""
    config: dict[str, object] = {
        "model": {"name": "sample", "version": "1"},
        "state_space": {"num_states": 2, "state_labels": ["left", "right"], "initial_state": 0},
        "observation_space": {"num_observations": 2, "observation_labels": ["dim", "bright"]},
        "action_space": {"num_actions": 2, "action_labels": ["hold", "switch"]},
        "matrices": {
            "A_matrix": {
                "shape": [2, 2],
                "initialization": "identity_based",
                "constraints": ["column_stochastic"],
            },
            "B_matrix": {
                "shape": [2, 2, 2],
                "initialization": "identity_based",
                "constraints": ["column_stochastic"],
            },
            "C_matrix": {
                "shape": [2],
                "initialization": "log_preferences",
                "initialization_params": {"preferences": [0.0, 1.0]},
            },
            "D_matrix": {"shape": [2], "initialization": "uniform"},
            "E_matrix": {"shape": [2], "initialization": "uniform"},
        },
        "inference": {"learning_rate": 0.5, "temperature": 1.0},
        "visualization": {"output_dir": "output"},
        "seed": 5,
    }
    config.update(overrides)
    return config


def test_valid_configuration_produces_stochastic_matrices() -> None:
    model = SimplePOMDP(pomdp_config())
    assert model.A.shape == (2, 2)
    assert np.allclose(model.A.sum(axis=0), 1.0)
    for action in range(2):
        assert np.allclose(model.B[:, :, action].sum(axis=0), 1.0)
    assert np.isclose(model.D.sum(), 1.0)
    assert np.isclose(model.E.sum(), 1.0)


def test_configuration_sections_are_checked_by_name() -> None:
    """Missing, unknown, and mistyped sections each produce their own message."""
    config = pomdp_config()
    del config["inference"]
    with pytest.raises(ValueError, match="Missing required configuration sections"):
        SimplePOMDP(config)

    with pytest.raises(ValueError, match="Unknown configuration sections"):
        SimplePOMDP(pomdp_config(surprise={"a": 1}))

    with pytest.raises(ValueError, match="model must be a mapping"):
        SimplePOMDP(pomdp_config(model=["not", "a", "mapping"]))

    with pytest.raises(ValueError, match="Unknown state_space fields"):
        SimplePOMDP(pomdp_config(state_space={"num_states": 2, "extra": 1}))

    config = pomdp_config()
    del config["matrices"]["E_matrix"]
    with pytest.raises(ValueError, match="must define exactly"):
        SimplePOMDP(config)

    config = pomdp_config()
    config["matrices"]["A_matrix"]["surprise"] = 1
    with pytest.raises(ValueError, match="Unknown A_matrix fields"):
        SimplePOMDP(config)

    config = pomdp_config()
    config["matrices"]["A_matrix"] = ["not a mapping"]
    with pytest.raises(ValueError, match="A_matrix must be a mapping"):
        SimplePOMDP(config)


def test_matrix_initialization_options_and_their_rejections() -> None:
    """Each documented initializer works; anything else is refused."""
    config = pomdp_config()
    config["matrices"]["A_matrix"]["initialization"] = "random_stochastic"
    assert np.allclose(SimplePOMDP(config).A.sum(axis=0), 1.0)

    config = pomdp_config()
    config["matrices"]["A_matrix"]["initialization"] = "spectral"
    with pytest.raises(ValueError, match="Unsupported initialization"):
        SimplePOMDP(config)

    config = pomdp_config()
    config["matrices"]["A_matrix"]["shape"] = [0, 2]
    with pytest.raises(ValueError, match="positive shape"):
        SimplePOMDP(config)

    config = pomdp_config()
    config["matrices"]["C_matrix"]["initialization_params"] = {"preferences": [0.0, 1.0, 2.0]}
    with pytest.raises(ValueError, match="preferences must have shape"):
        SimplePOMDP(config)

    config = pomdp_config()
    config["matrices"]["A_matrix"]["constraints"] = ["orthogonal"]
    with pytest.raises(ValueError, match="Unsupported matrix constraint"):
        SimplePOMDP(config)

    config = pomdp_config()
    config["matrices"]["B_matrix"]["constraints"] = ["row_stochastic"]
    with pytest.raises(ValueError, match="column_stochastic, not row_stochastic"):
        SimplePOMDP(config)

    config = pomdp_config()
    config["matrices"]["A_matrix"]["initialization"] = "identity_based"
    config["matrices"]["A_matrix"]["initialization_params"] = {"strength": 1.5}
    with pytest.raises(ValueError, match="identity strength"):
        SimplePOMDP(config)


def test_matrix_shapes_must_match_the_declared_spaces() -> None:
    config = pomdp_config()
    config["matrices"]["A_matrix"]["shape"] = [3, 2]
    with pytest.raises(ValueError, match="A matrix shape"):
        SimplePOMDP(config)

    config = pomdp_config()
    config["matrices"]["B_matrix"]["shape"] = [2, 2, 3]
    with pytest.raises(ValueError, match="B matrix shape"):
        SimplePOMDP(config)

    config = pomdp_config()
    config["matrices"]["C_matrix"]["shape"] = [3]
    config["matrices"]["C_matrix"]["initialization_params"] = {"preferences": [0.0, 1.0, 2.0]}
    with pytest.raises(ValueError, match="one preference per observation"):
        SimplePOMDP(config)

    config = pomdp_config()
    config["matrices"]["D_matrix"]["shape"] = [3]
    with pytest.raises(ValueError, match="one prior per state"):
        SimplePOMDP(config)

    config = pomdp_config()
    config["state_space"]["initial_state"] = 7
    with pytest.raises(ValueError, match="initial_state is out of range"):
        SimplePOMDP(config)


def test_configuration_file_paths_resolve_relative_to_the_file(tmp_path: Path) -> None:
    """A relative visualization directory is anchored to the configuration file."""
    config_file = tmp_path / "nested" / "model.yaml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(yaml.safe_dump(pomdp_config()), encoding="utf-8")
    model = SimplePOMDP(config_file)
    assert Path(model.config["visualization"]["output_dir"]).is_absolute()
    assert Path(model.config["visualization"]["output_dir"]).parent == config_file.parent.resolve()

    not_a_mapping = tmp_path / "list.yaml"
    not_a_mapping.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping"):
        SimplePOMDP(not_a_mapping)


@pytest.mark.parametrize("steps", [0, 3])
def test_run_and_persistence_round_trip(tmp_path: Path, steps: int) -> None:
    """A saved episode preserves the current state and every history column."""
    model = SimplePOMDP(pomdp_config(visualization={"output_dir": str(tmp_path / "plots")}))
    results = model.run(steps)
    assert len(results) == steps
    assert all(0 <= observation < 2 for observation, _ in results)

    saved = model.save_state(tmp_path / "state.yaml")
    beliefs = model.state.beliefs.copy()
    time_step = model.state.time_step
    current_state = model.state.current_state
    history = {key: np.asarray(values).copy() for key, values in model.state.history.items()}
    model.reset()
    assert model.state.time_step == 0
    model.load_state(saved)
    assert model.state.beliefs == pytest.approx(beliefs)
    assert model.state.time_step == time_step
    assert model.state.current_state == current_state
    assert set(model.state.history) == set(history)
    for key, values in history.items():
        np.testing.assert_allclose(model.state.history[key], values)

    with pytest.raises(ValueError, match="steps must be non-negative"):
        model.run(-1)
    with pytest.raises(ValueError, match="one action per step"):
        model.run(2, actions=[0])
    with pytest.raises(ValueError, match="Action out of range"):
        model.step(action=9)


def test_persistence_rejects_incompatible_saved_state(tmp_path: Path) -> None:
    model = SimplePOMDP(pomdp_config())
    saved = model.save_state(tmp_path / "state.yaml")
    payload = yaml.safe_load(saved.read_text(encoding="utf-8"))

    payload["schema_version"] = 2
    saved.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported or malformed POMDP state"):
        model.load_state(saved)

    payload["schema_version"] = 1
    payload["beliefs"] = [0.2, 0.3, 0.5]
    saved.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="dimensions are incompatible"):
        model.load_state(saved)

    payload["beliefs"] = [0.5, 0.5]
    payload["history"] = {"unexpected": []}
    saved.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="history fields are incompatible"):
        model.load_state(saved)


@pytest.mark.parametrize(
    "learning_rate",
    [None, "0.5", "invalid", [], {}, True, 1j, float("nan"), float("inf"), -float("inf")],
)
def test_learning_rate_is_rejected_before_model_construction(
    tmp_path: Path, learning_rate: object
) -> None:
    plots = tmp_path / "plots"
    with pytest.raises(ValueError, match=r"inference\.learning_rate"):
        SimplePOMDP(
            pomdp_config(
                inference={"learning_rate": learning_rate},
                visualization={"output_dir": str(plots)},
            )
        )
    assert not plots.exists()


@pytest.mark.parametrize("learning_rate", [-2.0, 0.0, 0.25, 1.0, 2.0, np.float32(0.25)])
def test_finite_learning_rates_keep_the_existing_clipped_bayesian_blend(
    tmp_path: Path, learning_rate: float
) -> None:
    model = SimplePOMDP(
        pomdp_config(
            inference={"learning_rate": learning_rate},
            visualization={"output_dir": str(tmp_path / "plots")},
        )
    )
    prior = model.state.beliefs.copy()
    predicted = model.B[:, :, 0] @ prior
    observation, _ = model.step(action=0)
    posterior = predicted * model.A[observation]
    posterior /= posterior.sum()
    weight = min(max(float(learning_rate), 0.0), 1.0)
    assert model.state.beliefs == pytest.approx((1.0 - weight) * prior + weight * posterior)


@pytest.mark.parametrize(
    "missing", ["schema_version", "current_state", "beliefs", "time_step", "history"]
)
def test_saved_state_requires_all_fields_without_mutating_live_state(
    tmp_path: Path, missing: str
) -> None:
    model = SimplePOMDP(pomdp_config(visualization={"output_dir": str(tmp_path / "plots")}))
    model.run(2, actions=[0, 1])
    saved = model.save_state(tmp_path / "state.yaml")
    original = saved.read_text(encoding="utf-8")
    original_state = model.state
    payload = yaml.safe_load(original)
    del payload[missing]
    saved.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        model.load_state(saved)

    assert model.state is original_state
    model.save_state(saved)
    assert saved.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("field_path", "invalid"),
    [
        (("schema_version",), True),
        (("schema_version",), 1.0),
        (("current_state",), -1),
        (("current_state",), 2),
        (("current_state",), 0.5),
        (("current_state",), True),
        (("current_state",), "1"),
        (("current_state",), float("nan")),
        (("time_step",), -1),
        (("time_step",), 0.5),
        (("time_step",), True),
        (("time_step",), "2"),
        (("time_step",), float("inf")),
        (("time_step",), float("nan")),
        (("time_step",), 0),
        (("time_step",), 3),
        (("beliefs",), 0.5),
        (("beliefs",), [[0.5, 0.5]]),
        (("beliefs",), [0.2, 0.3, 0.5]),
        (("beliefs",), ["0.5", 0.5]),
        (("beliefs",), [True, 0.5]),
        (("beliefs",), [float("nan"), 0.5]),
        (("beliefs",), [float("inf"), 0.5]),
        (("beliefs",), [-0.1, 1.1]),
        (("beliefs",), [0.0, 0.0]),
        (("history",), None),
        (("history",), {}),
        (("history", "states"), [0]),
        (("history", "free_energy"), [0.0, 0.0, 0.0]),
        (("history", "free_energy", 0), [0.0]),
        (("history", "free_energy", 0), "0.0"),
        (("history", "free_energy", 0), True),
        (("history", "free_energy", 0), float("nan")),
        (("history", "free_energy", 0), float("inf")),
        (("history", "beliefs", 0), [-0.1, 1.1]),
        (("history", "beliefs", 0), [0.0, 0.0]),
    ]
    + [
        (("history", key), invalid)
        for key in (
            "states",
            "observations",
            "actions",
            "beliefs",
            "free_energy",
            "efe_total",
            "efe_epistemic",
            "efe_pragmatic",
        )
        for invalid in ("00", None)
    ]
    + [
        (("history", key, 0), invalid)
        for key in ("states", "observations", "actions")
        for invalid in (-1, 2, 0.5, True, "1", float("nan"), float("inf"))
    ]
    + [
        (("history", key, 0), invalid)
        for key in ("beliefs", "efe_total", "efe_epistemic", "efe_pragmatic")
        for invalid in (
            0.5,
            [0.5],
            [[0.5, 0.5]],
            ["0.5", 0.5],
            [True, 0.5],
            [float("nan"), 0.5],
            [float("inf"), 0.5],
        )
    ],
)
def test_saved_state_rejection_preserves_the_complete_live_state(
    tmp_path: Path, field_path: tuple[str | int, ...], invalid: object
) -> None:
    model = SimplePOMDP(pomdp_config(visualization={"output_dir": str(tmp_path / "plots")}))
    model.run(2, actions=[0, 1])
    saved = model.save_state(tmp_path / "state.yaml")
    original = saved.read_text(encoding="utf-8")
    original_state = model.state
    payload = yaml.safe_load(original)
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = invalid
    saved.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        model.load_state(saved)

    assert model.state is original_state
    model.save_state(saved)
    assert saved.read_text(encoding="utf-8") == original


def test_malformed_saved_yaml_preserves_the_live_state(tmp_path: Path) -> None:
    model = SimplePOMDP(pomdp_config(visualization={"output_dir": str(tmp_path / "plots")}))
    model.run(2, actions=[0, 1])
    saved = model.save_state(tmp_path / "state.yaml")
    original = saved.read_text(encoding="utf-8")
    original_state = model.state
    saved.write_text("history: [", encoding="utf-8")
    with pytest.raises(ValueError):
        model.load_state(saved)
    assert model.state is original_state
    model.save_state(saved)
    assert saved.read_text(encoding="utf-8") == original


def test_saved_state_uses_distinct_space_dimensions_and_normalizes_positive_beliefs(
    tmp_path: Path,
) -> None:
    config = pomdp_config(visualization={"output_dir": str(tmp_path / "plots")})
    config["observation_space"] = {"num_observations": 3}
    config["action_space"] = {"num_actions": 4}
    config["matrices"]["A_matrix"] = {
        "shape": [3, 2],
        "initialization": "uniform",
        "constraints": ["column_stochastic"],
    }
    config["matrices"]["B_matrix"]["shape"] = [2, 2, 4]
    config["matrices"]["C_matrix"]["shape"] = [3]
    config["matrices"]["C_matrix"]["initialization_params"]["preferences"] = [0.0, 1.0, 2.0]
    config["matrices"]["E_matrix"]["shape"] = [4]
    model = SimplePOMDP(config)
    model.run(2, actions=[2, 3])
    saved = model.save_state(tmp_path / "state.yaml")
    payload = yaml.safe_load(saved.read_text(encoding="utf-8"))
    payload["current_state"] = 1.0
    payload["time_step"] = 2.0
    payload["beliefs"] = [2.0, 3.0]
    payload["history"]["states"] = [0.0, 1.0]
    payload["history"]["observations"] = [1.0, 2.0]
    payload["history"]["actions"] = [2.0, 3.0]
    payload["history"]["beliefs"][0] = [1e-200, 3e-200]
    saved.write_text(yaml.safe_dump(payload), encoding="utf-8")
    model.load_state(saved)
    assert model.state.current_state == 1 and model.state.time_step == 2
    assert model.state.beliefs == pytest.approx([0.4, 0.6])
    assert model.state.history["beliefs"][0] == pytest.approx([0.25, 0.75])
    assert model.state.history["observations"] == [1, 2]
    assert model.state.history["actions"] == [2, 3]
    assert np.asarray(model.state.history["efe_total"]).shape == (2, 4)
    assert np.asarray(model.state.history["efe_epistemic"]).shape == (2, 4)
    assert np.asarray(model.state.history["efe_pragmatic"]).shape == (2, 4)

    for key, out_of_range in (("states", 2), ("observations", 3), ("actions", 4)):
        model.save_state(saved)
        original = saved.read_text(encoding="utf-8")
        original_state = model.state
        payload = yaml.safe_load(original)
        payload["history"][key][0] = out_of_range
        saved.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            model.load_state(saved)
        assert model.state is original_state
        model.save_state(saved)
        assert saved.read_text(encoding="utf-8") == original


def _docs_tree(root: Path) -> None:
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "target.md").write_text("# Target\n\n## A Section\n", encoding="utf-8")


def test_wiki_link_gate_separates_broken_links_from_concepts(tmp_path: Path) -> None:
    """An unresolvable explicit file link is broken; a bare concept is not."""
    import subprocess

    _docs_tree(tmp_path)
    (tmp_path / "docs" / "page.md").write_text(
        "Resolves: [[target]]\n\nConcept only: [[an unwritten idea]]\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)

    lenient = verify_link_report(tmp_path, strict_wiki_links=False)
    assert lenient.broken_links == []
    assert lenient.skipped_concept_links >= 1
    assert lenient.resolved_links >= 1

    strict = verify_link_report(tmp_path, strict_wiki_links=True)
    assert strict.broken_links, "strict mode must report the unresolved concept"


def test_markdown_link_gate_reports_missing_files_and_anchors(tmp_path: Path) -> None:
    """File existence is an error; an unknown heading anchor is advisory."""
    import subprocess

    _docs_tree(tmp_path)
    (tmp_path / "docs" / "page.md").write_text(
        "[good](target.md)\n\n[anchor](target.md#a-section)\n\n"
        "[missing](absent.md)\n\n[bad anchor](target.md#not-a-heading)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)

    report = verify_markdown_link_report(tmp_path)
    assert report.checked_links >= 4
    assert any("absent.md" in str(entry) for entry in report.broken_links)
    assert not any("target.md#a-section" in str(entry) for entry in report.broken_links)
    assert any("not-a-heading" in str(entry) for entry in report.anchor_warnings)
