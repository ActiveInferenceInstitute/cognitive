"""Layered, validated settings for simulations, sweeps and reports.

The root ``config.yaml`` describes many things; most of it is documentation for
people. This module reads the three sections the runtime does consume and
rejects anything it cannot make sense of:

``active_inference``
    The inference parameters a simulation starts from. Each key maps to one
    :class:`~cognitive.models.active_inference.dispatcher.InferenceConfig`
    field (see :data:`INFERENCE_KEYS`).
``simulation``
    How an episode is run: the inference method, how policies sharing a first
    action are combined, the planning temperature, the number of steps and the
    random seed.
``output``
    Where figures and tables go and how they are written.

Values come from, in increasing priority: the built-in defaults, the YAML
file, ``COGNITIVE_<SECTION>__<KEY>`` environment variables, and explicit
``overrides`` such as ``{"simulation.steps": 50}``. Unknown keys inside a
section this module owns are errors, so a typo cannot silently fall back to a
default; other top-level sections of the file are ignored.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from .models.active_inference.dispatcher import InferenceConfig

DEFAULT_CONFIG_NAME = "config.yaml"
ENV_PREFIX = "COGNITIVE_"

# Keys of ``active_inference`` and the InferenceConfig field each one sets.
INFERENCE_KEYS: dict[str, str] = {
    "default_precision": "precision_init",
    "temporal_horizon": "temporal_horizon",
    "learning_rate": "learning_rate",
    "exploration_factor": "exploration_weight",
    "discount_factor": "discount_factor",
}

FIGURE_FORMATS = ("png", "svg", "pdf")


@dataclass(frozen=True)
class ActiveInferenceSettings:
    """Starting values for :class:`InferenceConfig`."""

    default_precision: float = 1.0
    temporal_horizon: int = 3
    learning_rate: float = 0.5
    exploration_factor: float = 0.0
    discount_factor: float = 0.95


@dataclass(frozen=True)
class SimulationSettings:
    """How an episode is run."""

    method: str = "variational"
    policy_aggregation: str = "marginal"
    temperature: float = 1.0
    policy_limit: int = 4096
    num_samples: int = 256
    steps: int = 20
    seed: int = 0


@dataclass(frozen=True)
class OutputSettings:
    """Where results are written and how figures are rendered."""

    directory: str = "output"
    figure_format: str = "png"
    dpi: int = 150


@dataclass(frozen=True)
class Settings:
    """All settings the runtime consumes, validated on construction."""

    active_inference: ActiveInferenceSettings = field(default_factory=ActiveInferenceSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    def __post_init__(self) -> None:
        if self.simulation.steps < 1:
            raise ValueError("simulation.steps must be at least one")
        if isinstance(self.simulation.seed, bool) or not isinstance(self.simulation.seed, int):
            raise ValueError("simulation.seed must be an integer")
        if self.output.figure_format not in FIGURE_FORMATS:
            raise ValueError(f"output.figure_format must be one of {list(FIGURE_FORMATS)}")
        if self.output.dpi < 1:
            raise ValueError("output.dpi must be positive")
        if not self.output.directory.strip():
            raise ValueError("output.directory must not be empty")
        # Building the config runs InferenceConfig's own validation, so a bad
        # learning rate or an unknown method fails here, not mid-simulation.
        self.inference_config()

    def inference_config(self, **overrides: Any) -> InferenceConfig:
        """Return the :class:`InferenceConfig` these settings describe."""
        values: dict[str, Any] = {
            target: getattr(self.active_inference, key) for key, target in INFERENCE_KEYS.items()
        }
        values.update(
            method=self.simulation.method,
            policy_type="discrete",
            policy_aggregation=self.simulation.policy_aggregation,
            temperature=self.simulation.temperature,
            policy_limit=self.simulation.policy_limit,
            num_samples=self.simulation.num_samples,
            seed=self.simulation.seed,
        )
        values.update(overrides)
        return InferenceConfig(**values)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """A plain nested mapping, suitable for JSON or YAML."""
        return {name: asdict(getattr(self, name)) for name in _SECTIONS}

    def with_overrides(self, overrides: Mapping[str, Any]) -> Settings:
        """Return a copy with ``{"section.key": value}`` overrides applied."""
        return _apply(self, _group(overrides, source="override"))


_SECTIONS: dict[str, type] = {
    "active_inference": ActiveInferenceSettings,
    "simulation": SimulationSettings,
    "output": OutputSettings,
}


def section_keys(section: str) -> list[str]:
    """The keys a section accepts, in declaration order."""
    return [item.name for item in fields(_SECTIONS[section])]


def _group(flat: Mapping[str, Any], *, source: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for dotted, value in flat.items():
        section, separator, key = str(dotted).partition(".")
        if not separator or section not in _SECTIONS:
            raise ValueError(
                f"{source} {dotted!r} must look like <section>.<key> "
                f"with a section in {sorted(_SECTIONS)}"
            )
        grouped.setdefault(section, {})[key] = value
    return grouped


def _coerce(section: str, key: str, value: Any, kind: Any) -> Any:
    """Convert ``value`` to the declared type, refusing lossy conversions."""
    if kind in (int, "int"):
        if isinstance(value, bool):
            raise ValueError(f"{section}.{key} must be an integer, got {value!r}")
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError as error:
                raise ValueError(f"{section}.{key} must be an integer, got {value!r}") from error
        if isinstance(value, float) and value == int(value):
            return int(value)
        if isinstance(value, int):
            return value
        raise ValueError(f"{section}.{key} must be an integer, got {value!r}")
    if kind in (float, "float"):
        if isinstance(value, bool):
            raise ValueError(f"{section}.{key} must be a number, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{section}.{key} must be a number, got {value!r}") from error
    if not isinstance(value, str):
        raise ValueError(f"{section}.{key} must be a string, got {value!r}")
    return value


def _apply(base: Settings, grouped: Mapping[str, Mapping[str, Any]]) -> Settings:
    updated: dict[str, Any] = {}
    for section, values in grouped.items():
        current = getattr(base, section)
        known = {item.name: item.type for item in fields(current)}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(
                f"Unknown {section} setting(s) {unknown}; valid keys are {sorted(known)}"
            )
        updated[section] = replace(
            current,
            **{key: _coerce(section, key, value, known[key]) for key, value in values.items()},
        )
    return replace(base, **updated)


def _from_environment(environ: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for name, value in environ.items():
        if not name.startswith(ENV_PREFIX) or "__" not in name:
            continue
        section, _, key = name[len(ENV_PREFIX) :].lower().partition("__")
        if section in _SECTIONS:
            grouped.setdefault(section, {})[key] = value
    return grouped


def read_config_file(path: str | Path) -> dict[str, dict[str, Any]]:
    """Return the sections of a YAML file that this module owns."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as error:
        raise FileNotFoundError(f"Cannot read configuration {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"{config_path} is not valid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a mapping at the top level")
    owned: dict[str, dict[str, Any]] = {}
    for section in _SECTIONS:
        if section not in raw:
            continue
        if not isinstance(raw[section], dict):
            raise ValueError(f"{config_path}: {section} must be a mapping")
        owned[section] = dict(raw[section])
    return owned


def load_settings(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Resolve settings from defaults, a YAML file, the environment and overrides.

    ``path=None`` uses ``config.yaml`` in the working directory when it exists
    and the built-in defaults otherwise; a path that is given must exist.
    ``environ`` defaults to ``os.environ``; pass ``{}`` to ignore it.
    """
    settings = Settings()
    if path is not None:
        settings = _apply(settings, read_config_file(path))
    elif Path(DEFAULT_CONFIG_NAME).is_file():
        settings = _apply(settings, read_config_file(DEFAULT_CONFIG_NAME))
    settings = _apply(settings, _from_environment(os.environ if environ is None else environ))
    if overrides:
        settings = settings.with_overrides(overrides)
    return settings
