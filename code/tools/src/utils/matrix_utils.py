"""Validated probability and expected-free-energy utilities."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _probability(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    total = array.sum(axis=0, keepdims=True)
    if np.any(total <= EPS):
        raise ValueError(f"{name} must contain positive mass")
    return array / total


def ensure_matrix_properties(
    matrix: np.ndarray,
    constraints: list[str] | str | None = None,
) -> np.ndarray:
    """Apply explicitly requested non-negativity and stochastic constraints."""
    values = np.asarray(matrix, dtype=float)
    requested = (
        []
        if constraints is None
        else [constraints]
        if isinstance(constraints, str)
        else list(constraints)
    )
    supported = {"column_stochastic", "row_stochastic", "non_negative"}
    unknown = set(requested).difference(supported)
    if unknown:
        raise ValueError(f"Unsupported matrix constraints: {sorted(unknown)}")
    if "non_negative" in requested:
        values = np.maximum(values, 0.0)
    if "column_stochastic" in requested and "row_stochastic" in requested:
        raise ValueError("Choose one stochastic orientation")
    if "column_stochastic" in requested:
        sums = values.sum(axis=0, keepdims=True)
        values = np.divide(
            values, sums, out=np.full_like(values, 1.0 / values.shape[0]), where=sums > EPS
        )
    elif "row_stochastic" in requested:
        sums = values.sum(axis=-1, keepdims=True)
        values = np.divide(
            values, sums, out=np.full_like(values, 1.0 / values.shape[-1]), where=sums > EPS
        )
    return values


def compute_entropy(distribution: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute Shannon entropy after normalizing along the requested axis."""
    probabilities = _probability(distribution, "distribution")
    return -np.sum(
        np.where(probabilities > 0, probabilities * np.log(probabilities + EPS), 0.0), axis=axis
    )


def softmax(x: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Compute a stable temperature-scaled softmax."""
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    values = np.asarray(x, dtype=float) / temperature
    if not np.all(np.isfinite(values)):
        raise ValueError("x must be finite")
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(np.clip(shifted, -745.0, 709.0))
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence between same-shaped probability vectors."""
    p_values = np.asarray(p, dtype=float).reshape(-1)
    q_values = np.asarray(q, dtype=float).reshape(-1)
    if p_values.shape != q_values.shape:
        raise ValueError("p and q must have the same shape")
    p_values = _probability(p_values, "p")
    q_values = _probability(q_values, "q")
    return float(
        np.sum(np.where(p_values > 0, p_values * np.log((p_values + EPS) / (q_values + EPS)), 0.0))
    )


def expected_free_energy(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    beliefs: np.ndarray,
    action: int,
) -> float:
    """Compute one-step risk plus ambiguity minus epistemic value."""
    likelihood = ensure_matrix_properties(
        np.asarray(A, dtype=float), ["column_stochastic", "non_negative"]
    )
    transition = np.asarray(B, dtype=float)
    if transition.ndim != 3 or not 0 <= int(action) < transition.shape[2]:
        raise ValueError("B must be (states, states, actions) and action must be valid")
    transitions = ensure_matrix_properties(
        transition[:, :, int(action)], ["column_stochastic", "non_negative"]
    )
    prior = _probability(np.asarray(beliefs, dtype=float).reshape(-1), "beliefs")
    predicted_states = transitions @ prior
    expected_obs = _probability(likelihood @ predicted_states, "expected observations").reshape(-1)
    preferences = softmax(np.asarray(C, dtype=float).reshape(-1))
    risk = kl_divergence(expected_obs, preferences)
    ambiguity = float(np.sum(expected_obs * np.log(1.0 / np.maximum(expected_obs, EPS))))
    return float(risk + ambiguity)


__all__ = [
    "compute_entropy",
    "ensure_matrix_properties",
    "expected_free_energy",
    "kl_divergence",
    "softmax",
]
