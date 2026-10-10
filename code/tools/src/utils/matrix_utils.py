"""Validated probability and expected-free-energy utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np

EPS = 1e-12


def as_integer(value: object, name: str) -> int:
    """Return ``value`` as an ``int`` or raise ``ValueError`` if it is not integral.

    ``int(1.9)`` is ``1``, so converting an index with ``int`` alone silently
    selects a different action or observation than the caller named. Integral
    floats such as ``1.0`` and NumPy integers are accepted; fractional,
    non-finite, and non-numeric values are rejected.
    """
    try:
        index = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if index != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return index


def normalize_mass(
    values: np.ndarray,
    axis: int = -1,
    *,
    empty: Literal["uniform", "zeros"] = "uniform",
) -> np.ndarray:
    """Divide non-negative ``values`` by their sum along ``axis``.

    The result is scale-invariant: ``normalize_mass(c * v)`` equals
    ``normalize_mass(v)`` for every ``c > 0``, however small. A fixed floor such
    as ``EPS = 1e-12`` on the total would instead treat a distribution whose
    entries are merely small as empty. Only a total that is not positive is
    empty, and ``empty`` chooses its replacement: ``"uniform"`` gives every
    entry ``1 / n`` along ``axis`` and ``"zeros"`` leaves the slice at zero.

    Callers validate that ``values`` is finite and non-negative; this function
    only normalizes. A total that overflows to infinity is rescaled by the
    largest entry first so that finite input never yields an all-zero result.
    """
    array = np.asarray(values, dtype=float)
    with np.errstate(over="ignore"):  # an overflowing total is handled just below
        totals = array.sum(axis=axis, keepdims=True)
    overflow = ~np.isfinite(totals)
    if np.any(overflow):
        peak = np.max(np.abs(array), axis=axis, keepdims=True)
        array = array / np.where(overflow & (peak > 0), peak, 1.0)
        totals = array.sum(axis=axis, keepdims=True)
    positive = totals > 0.0
    fill = 1.0 / array.shape[axis] if empty == "uniform" else 0.0
    return np.where(positive, array / np.where(positive, totals, 1.0), fill)


def _probability(values: np.ndarray, name: str, axis: int = 0) -> np.ndarray:
    """Normalize ``values`` into a probability distribution along ``axis``."""
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    if np.any(array.sum(axis=axis, keepdims=True) <= 0.0):
        raise ValueError(f"{name} must contain positive mass")
    return normalize_mass(array, axis)


def ensure_matrix_properties(
    matrix: np.ndarray,
    constraints: list[str] | str | None = None,
) -> np.ndarray:
    """Apply explicitly requested non-negativity and stochastic constraints.

    The input must be non-empty and finite: a NaN column would otherwise fail
    the ``sum > EPS`` test and be replaced by a uniform distribution, turning
    corrupt data into a plausible-looking result. Normalization is
    scale-invariant (see :func:`normalize_mass`): only a column or row whose
    entries sum to exactly zero becomes uniform. A stochastic constraint also
    requires non-negative entries; request ``"non_negative"`` as well to clip
    negative values first.
    """
    values = np.asarray(matrix, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("matrix must be non-empty and finite")
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
    if ({"column_stochastic", "row_stochastic"} & set(requested)) and np.any(values < 0):
        raise ValueError(
            "stochastic constraints require non-negative entries; "
            "add 'non_negative' to clip negative values first"
        )
    if "column_stochastic" in requested:
        values = normalize_mass(values, axis=0)
    elif "row_stochastic" in requested:
        values = normalize_mass(values, axis=-1)
    return values


def compute_entropy(distribution: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute Shannon entropy after normalizing along the requested axis.

    Normalization and summation share ``axis``, so entropies of a matrix's rows
    are computed from row-normalized values rather than from values normalized
    along some other axis. Zero-probability outcomes contribute nothing, so a
    deterministic distribution has entropy exactly zero rather than a value
    slightly below it.
    """
    probabilities = _probability(distribution, "distribution", axis=axis)
    return -np.sum(
        np.where(probabilities > 0, probabilities * np.log(np.maximum(probabilities, EPS)), 0.0),
        axis=axis,
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
    """Compute the one-step expected free energy ``risk + ambiguity``.

    ``risk`` is ``D_KL(q(o) || softmax(C))`` and ``ambiguity`` is the expected
    likelihood entropy ``E_q(s)[H(P(o | s))]`` -- the entropy of each *column*
    of ``A`` weighted by the predicted state distribution.  The entropy of the
    predicted observation ``H[q(o)]`` is a different quantity and is not the
    ambiguity term: it equals ``ambiguity + I(s; o)``.

    This returns the same value as
    ``DiscreteGenerativeModel.expected_free_energy(...)[0]`` for the same
    ``A``, ``B``, ``C``, beliefs, and action.
    """
    likelihood = ensure_matrix_properties(
        np.asarray(A, dtype=float), ["column_stochastic", "non_negative"]
    )
    transition = np.asarray(B, dtype=float)
    action_index = as_integer(action, "action")
    if transition.ndim != 3 or not 0 <= action_index < transition.shape[2]:
        raise ValueError("B must be (states, states, actions) and action must be valid")
    transitions = ensure_matrix_properties(
        transition[:, :, action_index], ["column_stochastic", "non_negative"]
    )
    prior = _probability(np.asarray(beliefs, dtype=float).reshape(-1), "beliefs")
    predicted_states = transitions @ prior
    expected_obs = _probability(likelihood @ predicted_states, "expected observations").reshape(-1)
    preferences = softmax(np.asarray(C, dtype=float).reshape(-1))
    risk = kl_divergence(expected_obs, preferences)
    ambiguity = float(np.sum(predicted_states * compute_entropy(likelihood, axis=0)))
    return float(risk + ambiguity)


__all__ = [
    "as_integer",
    "compute_entropy",
    "ensure_matrix_properties",
    "expected_free_energy",
    "kl_divergence",
    "normalize_mass",
    "softmax",
]
