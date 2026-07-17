"""Small reproducible runtime benchmarks with machine-readable output."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from cognitive.models.active_inference.base import ModelState
from cognitive.models.active_inference.dispatcher import ActiveInferenceDispatcher, InferenceConfig
from cognitive.models.active_inference.generative_model import DiscreteGenerativeModel
from cognitive.models.matrices.matrix_ops import MatrixOps
from Things.Continuous_Generic import ContinuousActiveInference
from Things.Simple_POMDP import SimplePOMDP


def _measure(
    name: str, function: Callable[[], None], repetitions: int
) -> dict[str, float | int | str]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    function()
    start = time.perf_counter()
    for _ in range(repetitions):
        function()
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "repetitions": repetitions,
        "total_seconds": elapsed,
        "seconds_per_call": elapsed / repetitions,
    }


def _model() -> DiscreteGenerativeModel:
    return DiscreteGenerativeModel(
        A=np.array([[0.9, 0.1], [0.1, 0.9]]),
        B=np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2),
        C=np.array([0.1, 1.0]),
        D=np.array([0.5, 0.5]),
        E=np.array([0.5, 0.5]),
    )


def _pomdp_config() -> dict[str, object]:
    return {
        "model": {"name": "benchmark", "version": "1"},
        "state_space": {"num_states": 2, "state_labels": ["left", "right"]},
        "observation_space": {"num_observations": 2, "observation_labels": ["left", "right"]},
        "action_space": {"num_actions": 2, "action_labels": ["stay", "switch"]},
        "matrices": {
            "A_matrix": {
                "shape": [2, 2],
                "initialization": "uniform",
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
        "visualization": {"output_dir": ".benchmarks"},
        "seed": 0,
    }


def run_benchmarks(repetitions: int = 10) -> dict[str, object]:
    """Run all supported benchmarks and return a JSON-serializable report."""
    model = _model()
    dispatcher = ActiveInferenceDispatcher(
        InferenceConfig(
            method="variational",
            policy_type="discrete",
            temporal_horizon=2,
            learning_rate=0.5,
            precision_init=1.0,
            seed=0,
        ),
        model,
    )
    state = ModelState(
        beliefs=model.D.copy(),
        policies=model.E.copy(),
        precision=1.0,
        free_energy=0.0,
        prediction_error=0.0,
    )
    matrix = np.random.default_rng(0).random((64, 64))
    pomdp = SimplePOMDP(_pomdp_config())
    continuous = ContinuousActiveInference(seed=0)
    benchmarks = [
        _measure("matrix_operations", lambda: MatrixOps.softmax(matrix, axis=0), repetitions),
        _measure(
            "dispatcher_inference", lambda: dispatcher.dispatch_belief_update(0, state), repetitions
        ),
        _measure("simple_pomdp", lambda: pomdp.step(action=0), repetitions),
        _measure("continuous_inference", lambda: continuous.step(np.zeros(2)), repetitions),
    ]
    return {"schema_version": 1, "repetitions": repetitions, "benchmarks": benchmarks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run cognitive runtime benchmarks")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, help="Write JSON to this path as well as stdout")
    args = parser.parse_args(argv)
    report = run_benchmarks(args.repetitions)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
