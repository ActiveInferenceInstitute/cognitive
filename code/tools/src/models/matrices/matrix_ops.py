"""Validated numerical operations for Active Inference probability matrices."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)
EPS = 1e-12


def _validate_numeric(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a non-empty finite numeric array")
    return values


def _normalize_axis(array: np.ndarray, axis: int) -> np.ndarray:
    values = np.maximum(_validate_numeric(array, "array"), 0.0)
    sums = values.sum(axis=axis, keepdims=True)
    count = values.shape[axis]
    return np.divide(
        values,
        np.where(sums > EPS, sums, 1.0),
        out=np.full_like(values, 1.0 / count),
        where=sums > EPS,
    )


class MatrixOps:
    """Core matrix operations with explicit probability contracts."""

    @staticmethod
    def normalize_columns(matrix: np.ndarray) -> np.ndarray:
        """Normalize non-negative values along the first axis."""
        return _normalize_axis(matrix, axis=0)

    @staticmethod
    def normalize_rows(matrix: np.ndarray) -> np.ndarray:
        """Normalize non-negative values along the last axis for 2-D matrices."""
        values = _validate_numeric(matrix, "matrix")
        if values.ndim != 2:
            raise ValueError("normalize_rows expects a two-dimensional matrix")
        return _normalize_axis(values, axis=1)

    @staticmethod
    def ensure_probability_distribution(matrix: np.ndarray) -> np.ndarray:
        """Clamp non-negative values and normalize columns."""
        return MatrixOps.normalize_columns(matrix)

    @staticmethod
    def compute_entropy(distribution: np.ndarray, axis: int | None = None) -> float | np.ndarray:
        """Compute Shannon entropy for a valid probability distribution."""
        values = _validate_numeric(distribution, "distribution")
        if np.any(values < 0):
            raise ValueError("distribution must be non-negative")
        sums = values.sum(axis=axis, keepdims=axis is not None)
        if not np.allclose(sums, 1.0):
            raise ValueError("distribution must sum to one")
        safe_values = np.where(values > 0, values, 1.0)
        terms = np.where(values > 0, values * np.log(safe_values), 0.0)
        result = -np.sum(terms, axis=axis)
        return float(result) if axis is None else result

    @staticmethod
    def compute_kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
        """Compute ``D_KL(P || Q)`` for same-shaped distributions."""
        p = _validate_numeric(P, "P")
        q = _validate_numeric(Q, "Q")
        if p.shape != q.shape:
            raise ValueError(f"Distributions must have same shape. Got P: {p.shape}, Q: {q.shape}")
        if (
            np.any(p < 0)
            or np.any(q < 0)
            or not np.isclose(p.sum(), 1.0)
            or not np.isclose(q.sum(), 1.0)
        ):
            raise ValueError("P and Q must be non-negative distributions that sum to one")
        return float(np.sum(np.where(p > 0, p * np.log((p + EPS) / (q + EPS)), 0.0)))

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
        """Apply numerically stable softmax along ``axis``."""
        if temperature <= 0 or not np.isfinite(temperature):
            raise ValueError("temperature must be finite and positive")
        values = _validate_numeric(x, "x") / temperature
        shifted = values - np.max(values, axis=axis, keepdims=True)
        exponentials = np.exp(np.clip(shifted, -745.0, 709.0))
        return exponentials / exponentials.sum(axis=axis, keepdims=True)


class MatrixLoader:
    """Load and validate matrix specifications and NumPy arrays."""

    @staticmethod
    def load_spec(spec_path: Path) -> dict[str, Any]:
        content = Path(spec_path).read_text(encoding="utf-8")
        if not content.startswith("---"):
            return {}
        _, _, remainder = content.partition("---")
        frontmatter, marker, _ = remainder.partition("---")
        if not marker:
            raise ValueError(f"Missing YAML frontmatter terminator: {spec_path}")
        parsed = yaml.safe_load(frontmatter) or {}
        if not isinstance(parsed, dict):
            raise ValueError("Matrix frontmatter must be a mapping")
        return parsed

    @staticmethod
    def load_matrix(data_path: Path) -> np.ndarray:
        return np.asarray(np.load(data_path, allow_pickle=False), dtype=float)

    @staticmethod
    def validate_matrix(matrix: np.ndarray, spec: dict[str, Any]) -> bool:
        values = np.asarray(matrix, dtype=float)
        if not np.all(np.isfinite(values)):
            return False
        dimensions = spec.get("dimensions")
        if dimensions:
            if "shape" in dimensions:
                expected_shape = tuple(int(value) for value in dimensions["shape"])
            else:
                expected_shape = tuple(int(dimensions[key]) for key in ("rows", "cols"))
            if values.shape != expected_shape:
                return False
        constraints = set(spec.get("shape_constraints", []))
        supported = {"sum(cols) == 1.0", "sum(rows) == 1.0", "all_values >= 0"}
        unknown = constraints.difference(supported)
        if unknown:
            raise ValueError(f"Unsupported matrix constraints: {sorted(unknown)}")
        if "all_values >= 0" in constraints and np.any(values < 0):
            return False
        if "sum(cols) == 1.0" in constraints and not np.allclose(values.sum(axis=0), 1.0):
            return False
        return not ("sum(rows) == 1.0" in constraints and not np.allclose(values.sum(axis=1), 1.0))


class MatrixInitializer:
    """Initialize finite, non-negative matrices reproducibly."""

    @staticmethod
    def random_stochastic(
        shape: tuple[int, ...], rng: np.random.Generator | None = None
    ) -> np.ndarray:
        if any(int(value) < 1 for value in shape):
            raise ValueError("shape dimensions must be positive")
        generator = rng or np.random.default_rng()
        return MatrixOps.normalize_columns(generator.random(shape))

    @staticmethod
    def identity_based(
        shape: tuple[int, ...],
        strength: float = 0.9,
    ) -> np.ndarray:
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 1:
            raise ValueError("identity_based requires a square, positive 2-D shape")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        size = shape[0]
        if size == 1:
            return np.ones(shape, dtype=float)
        matrix = np.full(shape, (1.0 - strength) / (size - 1), dtype=float)
        np.fill_diagonal(matrix, strength)
        return MatrixOps.normalize_columns(matrix)

    @staticmethod
    def uniform(shape: tuple[int, ...]) -> np.ndarray:
        if any(int(value) < 1 for value in shape):
            raise ValueError("shape dimensions must be positive")
        return np.full(shape, 1.0 / float(np.prod(shape)), dtype=float)


class MatrixVisualizer:
    """Small data adapters used by plotting code."""

    @staticmethod
    def prepare_heatmap_data(matrix: np.ndarray) -> dict[str, Any]:
        values = np.asarray(matrix)
        if values.ndim != 2:
            raise ValueError("heatmap data must be two-dimensional")
        return {
            "data": values,
            "x_ticks": range(values.shape[1]),
            "y_ticks": range(values.shape[0]),
        }

    @staticmethod
    def prepare_bar_data(vector: np.ndarray) -> dict[str, Any]:
        values = np.asarray(vector).reshape(-1)
        return {"data": values, "x_ticks": range(len(values))}

    @staticmethod
    def prepare_multi_heatmap_data(tensor: np.ndarray) -> dict[str, Any]:
        values = np.asarray(tensor)
        if values.ndim != 3:
            raise ValueError("multi-heatmap data must be three-dimensional")
        return {
            "slices": [values[index] for index in range(values.shape[0])],
            "x_ticks": range(values.shape[2]),
            "y_ticks": range(values.shape[1]),
        }
