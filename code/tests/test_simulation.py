"""Episodes, sweeps and model loading, checked against closed forms and invariants."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from cognitive.models.active_inference.dispatcher import InferenceConfig
from cognitive.settings import Settings
from cognitive.simulation import (
    SWEEPABLE,
    corridor_model,
    example_model,
    load_model,
    run_episode,
    save_table,
    summarize_sweep,
    sweep,
    sweep_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def config(**overrides: object) -> InferenceConfig:
    return Settings().inference_config(**overrides)


def test_an_episode_is_exactly_reproducible_from_its_seed() -> None:
    model = example_model()
    first = run_episode(model, config(), 25, seed=3)
    again = run_episode(model, config(), 25, seed=3)
    other = run_episode(model, config(), 25, seed=4)
    for name in ("beliefs", "states", "observations", "actions", "free_energy", "efe"):
        assert np.array_equal(getattr(first, name), getattr(again, name)), name
    assert not np.array_equal(first.actions, other.actions)


def test_seed_defaults_to_the_config_seed() -> None:
    model = example_model()
    assert run_episode(model, config(seed=6), 5).seed == 6
    assert run_episode(model, config(seed=None), 5).seed == 0


def test_the_agent_never_consumes_the_worlds_random_stream() -> None:
    model = example_model()
    steady = run_episode(model, config(temperature=0.2), 12, seed=9)
    noisy = run_episode(model, config(temperature=5.0, method="sampling"), 12, seed=9)
    assert steady.states[0] == noisy.states[0]  # the world's first draw is the same


def test_world_noise_does_not_depend_on_how_the_agent_samples() -> None:
    # Both actions leave the world the same, so only the world's own stream matters.
    indifferent = example_model(stickiness=0.5)
    exact = run_episode(indifferent, config(method="variational"), 25, seed=11)
    sampled = run_episode(indifferent, config(method="sampling", num_samples=64), 25, seed=11)
    assert np.array_equal(exact.states, sampled.states)
    assert np.array_equal(exact.observations, sampled.observations)
    assert not np.array_equal(exact.policy_posteriors, sampled.policy_posteriors)


def test_complexity_is_the_divergence_of_the_posterior_from_the_prediction() -> None:
    model = corridor_model()
    trajectory = run_episode(model, config(), 10, seed=3)
    for step in range(trajectory.steps):
        predicted = model.predict_states(trajectory.beliefs[step], int(trajectory.actions[step]))
        posterior = trajectory.beliefs[step + 1]
        expected = float(
            np.sum(posterior * np.log(np.maximum(posterior, 1e-12) / np.maximum(predicted, 1e-12)))
        )
        assert trajectory.complexity[step] == pytest.approx(expected, abs=1e-6)


def test_recorded_quantities_are_probabilities_and_obey_the_definitions() -> None:
    trajectory = run_episode(corridor_model(), config(), 30, seed=1)
    assert trajectory.beliefs.shape == (31, 4) and trajectory.states.shape == (31,)
    assert trajectory.policy_posteriors.shape == (30, 3) == trajectory.efe.shape
    assert np.allclose(trajectory.beliefs.sum(axis=1), 1.0)
    assert np.allclose(trajectory.policy_posteriors.sum(axis=1), 1.0)
    assert np.allclose(trajectory.free_energy, trajectory.complexity - trajectory.accuracy)
    assert np.all(trajectory.complexity >= -1e-12)
    assert np.allclose(trajectory.efe, trajectory.risk + trajectory.ambiguity)
    assert np.all(trajectory.epistemic >= 0)
    assert trajectory.actions.min() >= 0 and trajectory.actions.max() < 3
    assert trajectory.observations.max() < 4


def test_a_noiseless_sensor_and_full_learning_rate_make_beliefs_certain() -> None:
    model = example_model(sensor_noise=0.0)
    trajectory = run_episode(model, config(learning_rate=1.0), 20, seed=2)
    assert trajectory.belief_accuracy == pytest.approx(1.0)
    assert np.allclose(trajectory.beliefs[1:].max(axis=1), 1.0)


def test_initial_state_labels_and_the_summary_are_recorded() -> None:
    labels = {"states": ["dark", "light"], "actions": ["stay", "flip"], "observations": ["a", "b"]}
    trajectory = run_episode(example_model(), config(), 8, seed=0, initial_state=1, labels=labels)
    assert trajectory.states[0] == 1
    assert trajectory.state_labels == ("dark", "light")
    summary = trajectory.summary()
    assert summary["steps"] == 8 and sum(summary["action_counts"]) == 8
    assert 0.0 <= summary["preference_rate"] <= 1.0
    assert trajectory.config["policy_aggregation"] == "marginal"
    assert trajectory.preferred_observation == 1


def test_a_mismatched_world_is_allowed_when_the_shapes_agree() -> None:
    believed = example_model(stickiness=0.9)
    real = example_model(stickiness=0.55)
    right = run_episode(believed, config(), 60, seed=1)
    wrong = run_episode(believed, config(), 60, seed=1, process=real)
    assert wrong.belief_accuracy < right.belief_accuracy


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": 0}, "steps must be at least one"),
        ({"steps": 3, "initial_state": 2}, "initial_state out of range"),
        ({"steps": 3, "process": corridor_model()}, "same numbers of states"),
    ],
)
def test_episode_inputs_are_validated(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_episode(example_model(), config(), **kwargs)


def test_trajectories_serialize_to_json_and_csv(tmp_path: Path) -> None:
    trajectory = run_episode(example_model(), config(), 6, seed=5)
    payload = json.loads(trajectory.save_json(tmp_path / "a" / "t.json").read_text())
    assert payload["schema_version"] == 1 and payload["summary"]["steps"] == 6
    assert np.array(payload["beliefs"]).shape == (7, 2)
    with trajectory.save_csv(tmp_path / "b" / "t.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 and rows[0]["step"] == "0"
    assert float(rows[2]["free_energy"]) == pytest.approx(trajectory.free_energy[2])


def test_a_colder_agent_reaches_its_goal_more_often() -> None:
    model = corridor_model()
    cold = sweep(model, config(temperature=0.3), "temperature", [0.3], steps=30, seeds=range(8))
    warm = sweep(model, config(temperature=5.0), "temperature", [5.0], steps=30, seeds=range(8))
    cold_rate = summarize_sweep(cold)[0]["preference_rate"]
    warm_rate = summarize_sweep(warm)[0]["preference_rate"]
    assert cold_rate > warm_rate + 0.1


def test_sweep_runs_one_episode_per_value_and_seed_and_summarizes() -> None:
    rows = sweep(example_model(), config(), "temporal_horizon", [1, 2.0], steps=6, seeds=[0, 1, 2])
    assert len(rows) == 6 and {row.value for row in rows} == {1, 2}
    assert all(isinstance(row.value, int) for row in rows)
    summary = summarize_sweep(rows)
    assert [row["episodes"] for row in summary] == [3, 3]
    assert summary[0]["preference_rate_std"] >= 0.0
    flat = rows[0].flat()
    assert flat["parameter"] == "temporal_horizon" and "action_counts" not in flat


def test_sweeping_a_category_and_a_model_parameter() -> None:
    rows = sweep(
        example_model(), config(), "policy_aggregation", ["marginal", "min"], steps=5, seeds=[0]
    )
    assert [row.value for row in rows] == ["marginal", "min"]
    by_noise = sweep_model(
        lambda noise: example_model(sensor_noise=noise),
        "sensor_noise",
        [0.05, 0.4],
        config(),
        steps=40,
        seeds=range(6),
    )
    accuracy = {row["value"]: row["belief_accuracy"] for row in summarize_sweep(by_noise)}
    assert accuracy[0.05] > accuracy[0.4]


@pytest.mark.parametrize(
    ("parameter", "values", "message"),
    [
        ("bogus", [1], "Cannot sweep"),
        ("temperature", [], "values must not be empty"),
        ("temperature", [1.0, -1.0], "temperature must be finite and positive"),
        ("learning_rate", [0.5, 2.0], "learning_rate"),
        ("temporal_horizon", [2.5], "integer values"),
        ("temporal_horizon", [True], "integer values"),
        ("method", ["nonsense"], "nonsense"),
    ],
)
def test_sweeps_validate_every_value_before_running(
    parameter: str, values: list, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        sweep(example_model(), config(), parameter, values, steps=3)


def test_sweeps_need_seeds_and_the_sweepable_set_names_real_fields() -> None:
    with pytest.raises(ValueError, match="seeds must not be empty"):
        sweep(example_model(), config(), "temperature", [1.0], steps=3, seeds=[])
    with pytest.raises(ValueError, match="values must not be empty"):
        sweep_model(lambda value: example_model(), "x", [], config(), steps=3)
    with pytest.raises(ValueError, match="seeds must not be empty"):
        sweep_model(lambda value: example_model(), "x", [1.0], config(), steps=3, seeds=[])
    assert set(SWEEPABLE) <= {name for name in InferenceConfig.__dataclass_fields__}


def test_save_table_writes_a_header_and_rejects_nothing(tmp_path: Path) -> None:
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = save_table(rows, tmp_path / "t" / "rows.csv")
    assert path.read_text(encoding="utf-8").splitlines() == ["a,b", "1,x", "2,y"]
    with pytest.raises(ValueError, match="rows must not be empty"):
        save_table([], tmp_path / "none.csv")


def test_builtin_models_validate_their_arguments_and_are_stochastic() -> None:
    for build in (example_model, corridor_model):
        model = build()
        assert np.allclose(model.A.sum(axis=0), 1.0) and np.allclose(model.B.sum(axis=0), 1.0)
    with pytest.raises(ValueError, match="sensor_noise"):
        example_model(sensor_noise=0.7)
    with pytest.raises(ValueError, match="stickiness"):
        example_model(stickiness=0.2)
    with pytest.raises(ValueError, match="length"):
        corridor_model(length=1)
    with pytest.raises(ValueError, match="slip"):
        corridor_model(slip=1.0)
    with pytest.raises(ValueError, match="sensor_noise"):
        corridor_model(length=4, sensor_noise=0.8)
    walls = corridor_model(length=3, slip=0.0)
    assert walls.B[0, 0, 0] == 1.0 and walls.B[2, 2, 2] == 1.0  # walls stop the agent
    assert walls.D[0] == 1.0 and walls.preferences.argmax() == 2


def write_model(path: Path, **overrides: object) -> Path:
    content: dict[str, object] = {
        "A": [[0.8, 0.2], [0.2, 0.8]],
        "B_by_action": [[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]],
        "C": [0.0, 1.0],
        "D": [0.5, 0.5],
        "E": [0.5, 0.5],
        "labels": {"states": ["x", "y"]},
    }
    content.update(overrides)
    path.write_text(yaml.safe_dump({k: v for k, v in content.items() if v is not None}))
    return path


def test_load_model_reads_both_transition_layouts(tmp_path: Path) -> None:
    by_action, labels = load_model(write_model(tmp_path / "a.yaml"))
    standard = np.stack(
        [np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2
    )
    explicit, _ = load_model(
        write_model(tmp_path / "b.yaml", B_by_action=None, B=standard.tolist())
    )
    assert np.array_equal(by_action.B, explicit.B) and labels == {"states": ["x", "y"]}
    assert by_action.B[0, 1, 1] == pytest.approx(0.9)  # B[s_next, s_prev, a]


def test_the_shipped_model_file_loads_and_runs() -> None:
    model, labels = load_model(REPO_ROOT / "examples" / "configs" / "two_state.yaml")
    assert labels["actions"] == ["stay", "flip"]
    assert run_episode(model, config(), 5, seed=0).steps == 5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"extra": 1}, "unknown key"),
        ({"B": [[[1, 0], [0, 1]]]}, "exactly one of B and B_by_action"),
        ({"B_by_action": None}, "exactly one of B and B_by_action"),
        ({"labels": {"colours": []}}, "labels must map"),
        ({"labels": ["x"]}, "labels must map"),
    ],
)
def test_load_model_rejects_malformed_files(tmp_path: Path, overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_model(write_model(tmp_path / "m.yaml", **overrides))


def test_load_model_reports_unreadable_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Cannot read model file"):
        load_model(tmp_path / "absent.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("A: [", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_model(broken)
    broken.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_model(broken)


def test_replace_keeps_a_config_valid_for_a_run() -> None:
    assert run_episode(example_model(), replace(config(), temporal_horizon=1), 4, seed=0).steps == 4
