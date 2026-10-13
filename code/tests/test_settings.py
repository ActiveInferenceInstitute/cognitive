"""Settings: layering, validation and agreement with the shipped ``config.yaml``."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import yaml
from cognitive.models.active_inference.dispatcher import InferenceConfig
from cognitive.settings import (
    INFERENCE_KEYS,
    ActiveInferenceSettings,
    OutputSettings,
    Settings,
    SimulationSettings,
    load_settings,
    read_config_file,
    section_keys,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, content: dict) -> Path:
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_defaults_build_a_valid_inference_config() -> None:
    config = Settings().inference_config()
    assert config.temporal_horizon == 3
    assert config.policy_aggregation.value == "marginal"
    assert config.method.value == "variational"


def test_every_active_inference_key_sets_the_inference_field_it_names() -> None:
    settings = Settings(
        active_inference=ActiveInferenceSettings(
            default_precision=2.5,
            temporal_horizon=4,
            learning_rate=0.25,
            exploration_factor=0.2,
            discount_factor=0.8,
        ),
        simulation=SimulationSettings(temperature=0.7, policy_limit=99, num_samples=17, seed=5),
    )
    config = settings.inference_config()
    assert (config.precision_init, config.temporal_horizon, config.learning_rate) == (2.5, 4, 0.25)
    assert (config.exploration_weight, config.discount_factor) == (0.2, 0.8)
    assert (config.temperature, config.policy_limit, config.num_samples, config.seed) == (
        0.7,
        99,
        17,
        5,
    )
    assert set(INFERENCE_KEYS) == {item.name for item in fields(ActiveInferenceSettings)}
    inference_fields = {item.name for item in fields(InferenceConfig)}
    assert set(INFERENCE_KEYS.values()) <= inference_fields


def test_inference_config_accepts_explicit_overrides() -> None:
    assert Settings().inference_config(temporal_horizon=5).temporal_horizon == 5


def test_the_shipped_config_yaml_is_read_and_covers_every_key() -> None:
    owned = read_config_file(REPO_ROOT / "config.yaml")
    for section in ("active_inference", "simulation", "output"):
        assert set(owned[section]) == set(section_keys(section)), section
    assert load_settings(REPO_ROOT / "config.yaml", environ={}) == Settings()


def test_layers_apply_in_order_defaults_file_environment_overrides(tmp_path: Path) -> None:
    path = write(
        tmp_path / "c.yaml",
        {"simulation": {"steps": 11, "seed": 1}, "output": {"dpi": 90}, "logging": {"x": 1}},
    )
    settings = load_settings(
        path,
        environ={"COGNITIVE_SIMULATION__STEPS": "22", "COGNITIVE_OUTPUT__DPI": "75", "OTHER": "z"},
        overrides={"simulation.steps": 33},
    )
    assert settings.simulation.steps == 33  # override beats environment
    assert settings.output.dpi == 75  # environment beats file
    assert settings.simulation.seed == 1  # file beats default
    assert settings.active_inference.temporal_horizon == 3  # default survives


def test_without_a_path_the_working_directory_config_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_settings(environ={}).simulation.steps == 20
    write(tmp_path / "config.yaml", {"simulation": {"steps": 9}})
    assert load_settings(environ={}).simulation.steps == 9


def test_environment_is_read_from_the_process_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COGNITIVE_SIMULATION__SEED", "13")
    monkeypatch.setenv("COGNITIVE_UNKNOWN__KEY", "1")  # sections this module does not own: ignored
    assert load_settings().simulation.seed == 13


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"simulation.stepz": 3}, "Unknown simulation setting"),
        ({"steps": 3}, "must look like <section>.<key>"),
        ({"nothing.steps": 3}, "must look like <section>.<key>"),
        ({"simulation.steps": 0}, "at least one"),
        ({"simulation.steps": "many"}, "must be an integer"),
        ({"simulation.steps": True}, "must be an integer"),
        ({"simulation.steps": 2.5}, "must be an integer"),
        ({"simulation.seed": [1]}, "must be an integer"),
        ({"active_inference.learning_rate": "fast"}, "must be a number"),
        ({"active_inference.learning_rate": False}, "must be a number"),
        ({"active_inference.learning_rate": 2.0}, "learning_rate"),
        ({"simulation.method": 3}, "must be a string"),
        ({"simulation.method": "quantum"}, "quantum"),
        ({"output.figure_format": "gif"}, "figure_format"),
        ({"output.dpi": 0}, "dpi"),
        ({"output.directory": "  "}, "directory"),
    ],
)
def test_bad_values_are_rejected_with_the_reason(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Settings().with_overrides(overrides)


def test_whole_floats_are_accepted_as_integers_and_strings_are_parsed() -> None:
    settings = Settings().with_overrides({"simulation.steps": 7.0, "simulation.seed": "4"})
    assert (settings.simulation.steps, settings.simulation.seed) == (7, 4)


def test_a_seed_that_is_not_an_integer_cannot_be_constructed_directly() -> None:
    with pytest.raises(ValueError, match="seed must be an integer"):
        Settings(simulation=SimulationSettings(seed=1.5))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed must be an integer"):
        Settings(simulation=SimulationSettings(seed=True))


def test_unknown_keys_in_an_owned_section_of_the_file_are_errors(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", {"active_inference": {"learning_rat": 0.5}})
    with pytest.raises(ValueError, match="Unknown active_inference setting"):
        load_settings(path, environ={})


@pytest.mark.parametrize(
    "content",
    [["not", "a", "mapping"], {"simulation": "steps: 3"}],
)
def test_file_shape_errors(tmp_path: Path, content: object) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_settings(path, environ={})


def test_missing_and_malformed_files_are_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Cannot read configuration"):
        load_settings(tmp_path / "absent.yaml", environ={})
    broken = tmp_path / "broken.yaml"
    broken.write_text("simulation: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_settings(broken, environ={})


def test_an_empty_file_means_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_settings(path, environ={}) == Settings()


def test_to_dict_round_trips_through_overrides() -> None:
    original = Settings().with_overrides({"simulation.steps": 5, "output.dpi": 80})
    flat = {
        f"{section}.{key}": value
        for section, values in original.to_dict().items()
        for key, value in values.items()
    }
    assert Settings().with_overrides(flat) == original
    assert OutputSettings().figure_format == "png"
