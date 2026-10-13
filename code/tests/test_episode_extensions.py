"""Closed-form episodes, saved-data rejection controls, and real CLI outputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from cognitive.cli import main
from cognitive.episodes import (
    Hierarchy,
    corridor_hierarchy,
    run_continuous_episode,
    run_hierarchical_episode,
)
from cognitive.models.active_inference.dispatcher import ActiveInferenceFactory
from cognitive.settings import Settings
from cognitive.simulation import Trajectory, corridor_model, run_episode
from Things.Continuous_Generic import ContinuousActiveInference


def test_trajectory_round_trip_preserves_arrays_and_recomputes_summary(tmp_path: Path) -> None:
    trajectory = run_episode(corridor_model(), Settings().inference_config(), 7, seed=3)
    data = trajectory.to_dict()
    data["summary"] = {"steps": 999}
    restored = Trajectory.from_dict(data)
    assert restored.summary() == trajectory.summary()
    for name in ("states", "actions", "beliefs", "efe", "free_energy"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(trajectory, name))
    saved = restored.save_json(tmp_path / "episode.json")
    assert Trajectory.load_json(saved).to_dict() == trajectory.to_dict()


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"seed": -1},
        {"seed": True},
        {"preferred_observation": 1.5},
        {"config": []},
        {"config": {"temperature": float("nan")}},
        {"labels": []},
        {"unknown": 0},
        {"beliefs": [[1, 0]]},
        {"beliefs": [[0.4, 0.4]] * 4},
        {"beliefs": [[-1, 2]] * 4},
        {"beliefs": [[float("nan"), 0]] * 4},
        {"beliefs": [["0.5", "0.5"]] * 4},
        {"beliefs": [[[], 1]]},
        {"policy_posteriors": []},
        {"risk": [0] * 3},
        {"states": [0, 1]},
        {"actions": [0, 1.5, 0]},
        {"actions": [0, 99, 0]},
        {"actions": [0, True, 0]},
        {"observations": [-1, 1, 1]},
        {"observations": [1e100, 1, 1]},
        {"free_energy": [999] * 3},
        {"efe": [[0, 0, 0]] * 3},
        {"complexity": [-1] * 3, "accuracy": [-1] * 3, "free_energy": [0] * 3},
        {"epistemic": [[-1, 0, 0]] * 3},
        {"labels": {"extra": []}},
        {"labels": {"states": [1, 2, 3, 4]}},
        {"labels": {"states": ["one"]}},
        {"labels": {"observations": ["one"]}, "observations": [2, 2, 2]},
    ],
)
def test_trajectory_parser_rejects_corruption(change: dict) -> None:
    data = run_episode(corridor_model(), Settings().inference_config(), 3).to_dict()
    data.update(change)
    with pytest.raises(ValueError):
        Trajectory.from_dict(data)


def test_trajectory_loader_rejects_missing_and_invalid_json(tmp_path: Path) -> None:
    data = run_episode(corridor_model(), Settings().inference_config(), 3).to_dict()
    del data["beliefs"]
    with pytest.raises(ValueError, match="Missing"):
        Trajectory.from_dict(data)
    with pytest.raises(FileNotFoundError):
        Trajectory.load_json(tmp_path / "absent.json")
    for text in ("[", "[]"):
        path = tmp_path / "bad.json"
        path.write_text(text)
        with pytest.raises(ValueError):
            Trajectory.load_json(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError):
        Trajectory.load_json(path)


def test_integer_observation_indices_are_not_rounded_through_float() -> None:
    data = run_episode(corridor_model(), Settings().inference_config(), 1).to_dict()
    data["observations"] = [2**53 + 1]
    loaded = Trajectory.from_dict(data)
    assert int(loaded.observations[0]) == 2**53 + 1


def test_continuous_world_and_beliefs_match_the_scalar_update() -> None:
    agent = ContinuousActiveInference(n_states=1, n_obs=1, n_orders=1, dt=0.1, alpha=0.5)
    trajectory = run_continuous_episode(agent, 2, initial_state=np.array([2.0]))
    np.testing.assert_allclose(trajectory.states[:, 0], [2.0, 1.8, 1.44])
    np.testing.assert_allclose(trajectory.belief_means[:, 0, 0], [0.0, 0.09, 0.1575])
    np.testing.assert_allclose(trajectory.actions[:, 0], [1.8, 1.35])
    np.testing.assert_allclose(trajectory.free_energy, 0.5 * np.array([1.71, 1.2825]) ** 2)
    np.testing.assert_allclose(trajectory.time, [0.0, 0.1, 0.2])
    assert agent.state.time == pytest.approx(0.2)


def test_continuous_noisy_episode_is_reproducible_and_serializes(tmp_path: Path) -> None:
    first = run_continuous_episode(
        ContinuousActiveInference(n_obs=1), 5, seed=2, process_noise=0.1, sensor_noise=0.2
    )
    again = run_continuous_episode(
        ContinuousActiveInference(n_obs=1), 5, seed=2, process_noise=0.1, sensor_noise=0.2
    )
    other = run_continuous_episode(
        ContinuousActiveInference(n_obs=1), 5, seed=3, process_noise=0.1, sensor_noise=0.2
    )
    np.testing.assert_array_equal(first.states, again.states)
    assert not np.array_equal(first.observations, other.observations)
    data = json.loads(first.save_json(tmp_path / "c.json").read_text())
    assert data["kind"] == "continuous" and data["config"]["n_obs"] == 1
    assert data["config"]["state_precision"] == [1.0, 1.0]
    assert np.shape(data["config"]["initial_belief_precisions"]) == (2, 3)
    assert len(first.save_csv(tmp_path / "c.csv").read_text().splitlines()) == 6


def test_continuous_episode_advances_from_existing_state() -> None:
    agent = ContinuousActiveInference(n_orders=1)
    agent.step(np.ones(2))
    initial = agent.state.belief_means.copy()
    trajectory = run_continuous_episode(agent, 1, process_dynamics=np.zeros((2, 2)))
    np.testing.assert_array_equal(trajectory.belief_means[0], initial)
    assert trajectory.time[0] == pytest.approx(0.01)
    np.testing.assert_allclose(trajectory.states, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"steps": True},
        {"seed": -1},
        {"seed": True},
        {"sensor_noise": -1},
        {"process_noise": float("nan")},
        {"control_gain": -1},
        {"initial_state": np.ones(3)},
        {"process_dynamics": np.ones((1, 2))},
    ],
)
def test_continuous_inputs_fail_before_advancing_agent(kwargs: dict) -> None:
    agent = ContinuousActiveInference()
    with pytest.raises(ValueError):
        run_continuous_episode(agent, **{"steps": 3, **kwargs})
    assert agent.state.time == 0


def test_hierarchical_episode_preserves_continuity_and_partial_final_interval(
    tmp_path: Path,
) -> None:
    config = Settings().inference_config(temporal_horizon=1)
    hierarchy = corridor_hierarchy()
    trajectory = run_hierarchical_episode(
        hierarchy, config, config, 12, decision_interval=5, seed=7
    )
    again = run_hierarchical_episode(hierarchy, config, config, 12, decision_interval=5, seed=7)
    np.testing.assert_array_equal(trajectory.decision_steps, [0, 5, 10])
    assert trajectory.lower.beliefs.shape == (13, 4)
    assert trajectory.upper_beliefs.shape == (4, 2)
    np.testing.assert_allclose(trajectory.upper_beliefs.sum(axis=1), 1)
    np.testing.assert_allclose(trajectory.upper_posteriors.sum(axis=1), 1)
    np.testing.assert_array_equal(trajectory.lower.states, again.lower.states)
    np.testing.assert_array_equal(trajectory.goals, again.goals)
    np.testing.assert_allclose(
        trajectory.lower.free_energy, trajectory.lower.complexity - trajectory.lower.accuracy
    )
    payload = json.loads(trajectory.save_json(tmp_path / "h.json").read_text())
    assert payload["lower"]["summary"]["steps"] == 12
    assert payload["upper_config"]["decision_interval"] == 5
    assert payload["upper_model"]["C"] == [0, 2]
    assert hierarchy.lower.D[0] == 1 and config.seed == 0
    model = hierarchy.lower
    for t in range(trajectory.lower.steps):
        predicted = model.predict_states(
            trajectory.lower.beliefs[t], int(trajectory.lower.actions[t])
        )
        posterior = model.posterior(int(trajectory.lower.observations[t]), predicted)
        expected = (1 - config.learning_rate) * predicted + config.learning_rate * posterior
        np.testing.assert_allclose(trajectory.lower.beliefs[t + 1], expected, atol=1e-10)


def test_upper_goal_selection_changes_lower_behavior() -> None:
    hierarchy = corridor_hierarchy()
    right = replace(hierarchy, upper=replace(hierarchy.upper, C=np.array([0.0, 20.0])))
    left = replace(hierarchy, upper=replace(hierarchy.upper, C=np.array([20.0, 0.0])))
    config = Settings().inference_config(temporal_horizon=2, temperature=0.1)
    toward_right = run_hierarchical_episode(right, config, config, 10)
    toward_left = run_hierarchical_episode(left, config, config, 10)
    assert np.all(toward_right.goals == 1) and np.all(toward_left.goals == 0)
    assert toward_right.lower.states[-1] > toward_left.lower.states[-1]


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"decision_interval": 0}, {"seed": -1}])
def test_hierarchical_runner_rejects_invalid_limits(kwargs: dict) -> None:
    config = Settings().inference_config()
    with pytest.raises(ValueError):
        run_hierarchical_episode(corridor_hierarchy(), config, config, **{"steps": 3, **kwargs})


@pytest.mark.parametrize(
    "goals,mapping",
    [
        (np.zeros((2, 2)), np.array([0, 0, 1, 1])),
        (np.full((2, 4), np.nan), np.array([0, 0, 1, 1])),
        (np.zeros((2, 4)), np.array([0, 0, 1, 2])),
        (np.zeros((2, 4)), np.array([0.0, 0.0, 1.0, 1.0])),
        (np.zeros((2, 4)), np.array([0, 1])),
    ],
)
def test_hierarchy_rejects_invalid_cross_level_mapping(
    goals: np.ndarray, mapping: np.ndarray
) -> None:
    hierarchy = corridor_hierarchy()
    with pytest.raises(ValueError):
        Hierarchy(hierarchy.lower, hierarchy.upper, goals, mapping)


@pytest.mark.parametrize("command", ["continuous", "hierarchical"])
def test_episode_cli_writes_real_data_and_figure(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command, "--steps", "7", "--seed", "3", "--output", str(tmp_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert all(Path(name).stat().st_size > 0 for name in data["files"])
    assert (tmp_path / "episode.png").is_file()


def test_corridor_cli_and_saved_plot_use_recorded_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "simulate",
                "--builtin",
                "corridor",
                "--steps",
                "4",
                "--no-figures",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    path = tmp_path / "trajectory.json"
    original = path.read_bytes()
    assert len(json.loads(original)["beliefs"][0]) == 4
    assert (
        main(
            [
                "plot",
                str(path),
                "--step",
                "2",
                "--output",
                str(tmp_path / "plots"),
                "--set",
                "output.figure_format=svg",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert len(result["files"]) == 2 and (tmp_path / "plots" / "efe_step2.svg").is_file()
    assert path.read_bytes() == original
    assert main(["plot", str(path), "--step", "99"]) == 2
    assert "error:" in capsys.readouterr().err


def test_model_selector_options_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as error:
        main(["simulate", "--builtin", "corridor", "--model", "any.yaml"])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["continuous", "hierarchical"])
def test_episode_cli_skips_figures_when_requested(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command, "--steps", "2", "--no-figures", "--output", str(tmp_path)]) == 0
    assert "trajectory.json" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.png"))


def test_continuous_nonfinite_integration_is_rejected() -> None:
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError):
        run_continuous_episode(
            ContinuousActiveInference(n_orders=1), 2, initial_state=np.full(2, 1e200)
        )


def test_removed_configuration_paths_are_rejected(tmp_path: Path) -> None:
    for key in ("model", "custom_params"):
        document = {
            "method": "variational",
            "policy_type": "discrete",
            "temporal_horizon": 1,
            "learning_rate": 0.5,
            "precision_init": 1.0,
            "generative_model": {
                name: value.tolist() for name, value in corridor_model().__dict__.items()
            },
        }
        if key == "model":
            document[key] = document.pop("generative_model")
        else:
            document[key] = {"unused": 1}
        path = tmp_path / f"{key}.yaml"
        path.write_text(yaml.safe_dump(document))
        with pytest.raises(ValueError):
            ActiveInferenceFactory.create_from_yaml(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"B_by_action": 3},
        {"B_by_action": []},
        {"labels": []},
        {"labels": {"actions": "abc"}},
        {"labels": {"states": ["only"]}},
        {"labels": {"observations": [1, 2, 3, 4]}},
        {3: 1},
    ],
)
def test_model_cli_rejects_invalid_yaml_without_a_traceback(
    changes: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = corridor_model()
    document = {name: value.tolist() for name, value in model.__dict__.items() if name != "B"}
    document["B_by_action"] = np.moveaxis(model.B, 2, 0).tolist()
    document.update(changes)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document))
    assert main(["simulate", "--model", str(path), "--steps", "3", "--no-figures"]) == 2
    assert capsys.readouterr().err.startswith("error:")
