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


def test_run_and_persistence_round_trip(tmp_path: Path) -> None:
    """A simulation replays from disk with its history intact."""
    model = SimplePOMDP(pomdp_config())
    results = model.run(3)
    assert len(results) == 3
    assert all(0 <= observation < 2 for observation, _ in results)

    saved = model.save_state(tmp_path / "state.yaml")
    beliefs = model.state.beliefs.copy()
    time_step = model.state.time_step
    model.reset()
    assert model.state.time_step == 0
    model.load_state(saved)
    assert model.state.beliefs == pytest.approx(beliefs)
    assert model.state.time_step == time_step

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
