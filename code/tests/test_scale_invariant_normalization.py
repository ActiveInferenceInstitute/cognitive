"""Normalization must not depend on the scale of the input.

Every normalizer here compared the total mass with ``EPS = 1e-12`` and treated
anything smaller as empty, so ``[1e-13, 3e-13]`` became uniform (or raised)
while ``[1e-11, 3e-11]`` became ``[0.25, 0.75]``. Dividing by the sum is
scale-invariant; only a total of exactly zero is empty.
"""

from __future__ import annotations

import numpy as np
import pytest
from cognitive.models.active_inference.base import ActiveInferenceModel, ModelState
from cognitive.models.active_inference.dispatcher import ActiveInferenceDispatcher, InferenceConfig
from cognitive.models.active_inference.generative_model import (
    DiscreteGenerativeModel,
    _as_probability_vector,
    _normalize,
)
from cognitive.models.active_inference.homeostatic import HomeostaticInference
from cognitive.models.matrices.matrix_ops import MatrixOps
from cognitive.utils.matrix_utils import (
    compute_entropy,
    ensure_matrix_properties,
    kl_divergence,
    normalize_mass,
)
from Things.Simple_POMDP.simple_pomdp import SimplePOMDP
from Things.Simple_POMDP.simple_pomdp import _normalize as pomdp_normalize

SCALES = [1.0, 1e-11, 1e-13, 1e-20, 1e-150, 1e-300]
SHAPE = np.array([1.0, 3.0])
EXPECTED = np.array([0.25, 0.75])


@pytest.mark.parametrize("scale", SCALES)
def test_normalize_mass_is_scale_invariant(scale: float) -> None:
    assert normalize_mass(SHAPE * scale) == pytest.approx(EXPECTED, rel=1e-12)


@pytest.mark.parametrize("scale", [1e100, 1e200, 1e308])
def test_normalize_mass_survives_a_total_that_overflows(scale: float) -> None:
    result = normalize_mass(np.array([scale, scale, 2 * scale if scale < 1e308 else scale]))
    assert np.isfinite(result).all()
    assert result.sum() == pytest.approx(1.0)


def test_normalize_mass_is_exact_for_ordinary_input() -> None:
    """Inputs with mass above the old floor give exactly the plain quotient."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        values = rng.random((4, 3)) + 0.01
        assert np.array_equal(normalize_mass(values, axis=0), values / values.sum(0, keepdims=True))
        assert np.array_equal(normalize_mass(values, axis=1), values / values.sum(1, keepdims=True))


def test_normalize_mass_zero_slices_follow_the_requested_fallback() -> None:
    values = np.array([[0.0, 2.0], [0.0, 6.0], [0.0, 0.0]])
    uniform = normalize_mass(values, axis=0)
    assert uniform[:, 0] == pytest.approx(np.full(3, 1 / 3))
    assert uniform[:, 1] == pytest.approx([0.25, 0.75, 0.0])
    zeros = normalize_mass(values, axis=0, empty="zeros")
    assert zeros[:, 0] == pytest.approx(np.zeros(3))
    assert zeros[:, 1] == pytest.approx([0.25, 0.75, 0.0])


def _dispatcher() -> ActiveInferenceDispatcher:
    model = DiscreteGenerativeModel(
        A=np.eye(2), B=np.stack([np.eye(2)] * 2, axis=2), C=np.zeros(2),
        D=np.full(2, 0.5), E=np.full(2, 0.5),
    )  # fmt: skip
    return ActiveInferenceDispatcher(InferenceConfig("variational", "discrete", 1, 0.5, 1.0), model)


@pytest.mark.parametrize("scale", SCALES)
def test_dispatcher_normalize(scale: float) -> None:
    assert _dispatcher()._normalize(SHAPE * scale, 2) == pytest.approx(EXPECTED)


@pytest.mark.parametrize("scale", SCALES)
def test_homeostatic_normalize(scale: float) -> None:
    assert HomeostaticInference._normalize(SHAPE * scale, 2) == pytest.approx(EXPECTED)


@pytest.mark.parametrize("scale", SCALES)
def test_matrix_ops_normalize(scale: float) -> None:
    matrix = np.outer(SHAPE, [1.0, 2.0]) * scale
    columns = MatrixOps.normalize_columns(matrix)
    assert columns[:, 0] == pytest.approx(EXPECTED)
    assert columns[:, 1] == pytest.approx(EXPECTED)
    assert MatrixOps.normalize_rows(matrix.T)[0] == pytest.approx(EXPECTED)


@pytest.mark.parametrize("scale", SCALES)
def test_ensure_matrix_properties_column_and_row(scale: float) -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 6.0]]) * scale
    column = ensure_matrix_properties(matrix, "column_stochastic")
    assert column[:, 0] == pytest.approx(EXPECTED)
    row = ensure_matrix_properties(matrix, "row_stochastic")
    assert row[0] == pytest.approx([1 / 3, 2 / 3])


@pytest.mark.parametrize("scale", SCALES)
def test_probability_helpers_accept_small_distributions(scale: float) -> None:
    """Entropy and KL used to raise ``must contain positive mass`` below 1e-12."""
    reference = compute_entropy(SHAPE)
    assert compute_entropy(SHAPE * scale) == pytest.approx(reference)
    assert kl_divergence(SHAPE * scale, np.ones(2) * scale) == pytest.approx(
        kl_divergence(SHAPE, np.ones(2)), abs=1e-9
    )


@pytest.mark.parametrize("scale", SCALES)
def test_generative_model_normalizers(scale: float) -> None:
    assert _normalize(SHAPE * scale) == pytest.approx(EXPECTED)
    assert _as_probability_vector(SHAPE * scale, name="v") == pytest.approx(EXPECTED)
    model = DiscreteGenerativeModel(
        A=np.eye(2), B=np.stack([np.eye(2)] * 2, axis=2), C=np.zeros(2),
        D=SHAPE * scale, E=SHAPE * scale,
    )  # fmt: skip
    assert pytest.approx(EXPECTED) == model.D
    assert pytest.approx(EXPECTED) == model.E


@pytest.mark.parametrize("scale", SCALES)
def test_simple_pomdp_normalizers(scale: float) -> None:
    assert pomdp_normalize(SHAPE * scale) == pytest.approx(EXPECTED)
    matrix = np.outer(SHAPE, [1.0, 2.0]) * scale
    assert SimplePOMDP._column_normalize(matrix)[:, 1] == pytest.approx(EXPECTED)
    assert SimplePOMDP._row_normalize(matrix.T)[1] == pytest.approx(EXPECTED)


def test_simple_pomdp_all_zero_columns_stay_zero() -> None:
    """The documented fallback is unchanged: an all-zero slice is not invented."""
    zeros = np.zeros((2, 2))
    assert np.array_equal(SimplePOMDP._column_normalize(zeros), zeros)
    assert np.array_equal(SimplePOMDP._row_normalize(zeros), zeros)
    assert pomdp_normalize(np.zeros(4)) == pytest.approx(np.full(4, 0.25))


def test_all_zero_input_keeps_its_previous_handling() -> None:
    assert _dispatcher()._normalize(np.zeros(2), 2) == pytest.approx([0.5, 0.5])
    assert HomeostaticInference._normalize(np.zeros(3), 3) == pytest.approx(np.full(3, 1 / 3))
    assert MatrixOps.normalize_columns(np.zeros((2, 2))) == pytest.approx(np.full((2, 2), 0.5))
    assert ensure_matrix_properties(np.zeros((2, 2)), "column_stochastic") == pytest.approx(
        np.full((2, 2), 0.5)
    )
    assert _normalize(np.zeros(2)) == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError, match="positive mass"):
        _as_probability_vector(np.zeros(2), name="v")
    with pytest.raises(ValueError, match="positive mass"):
        compute_entropy(np.zeros(2))


def test_non_finite_input_is_still_rejected() -> None:
    for bad in (np.array([np.nan, 1.0]), np.array([np.inf, 1.0])):
        with pytest.raises(ValueError):
            _dispatcher()._normalize(bad, 2)
        with pytest.raises(ValueError):
            HomeostaticInference._normalize(bad, 2)
        with pytest.raises(ValueError):
            ensure_matrix_properties(bad.reshape(2, 1), "column_stochastic")
        with pytest.raises(ValueError):
            MatrixOps.normalize_columns(bad.reshape(2, 1))


class _Passive(ActiveInferenceModel):
    """Smallest concrete model, for the base-class normalizers."""

    def _load_config(self, config_path):
        return dict(config_path)

    def _initialize_matrices(self) -> None:
        pass

    def _initialize_state(self) -> ModelState:
        return ModelState(np.array([0.5, 0.5]), np.array([0.5, 0.5]), 1.0, 0.0, 0.0)

    def step(self, action=None):
        return 0, 0.0

    def visualize(self, plot_type, **kwargs):
        return plot_type


@pytest.mark.parametrize("scale", SCALES)
def test_base_model_normalizers(scale: float) -> None:
    model = _Passive({"generative_matrix": np.array([[1.0, 3.0], [1.0, 3.0]]) * scale})
    assert model._observation_vector(SHAPE * scale, 2) == pytest.approx(EXPECTED)
    model.state.beliefs = np.array([1.0, 0.0])
    assert model._generate_prediction() == pytest.approx(EXPECTED)
    model.config["generative_matrix"] = np.zeros((2, 2))
    assert model._generate_prediction() == pytest.approx([0.5, 0.5])

    model.config["prior_beliefs"] = SHAPE * scale
    model.state.beliefs = SHAPE * scale
    model.state.prediction_error = 0.0
    assert model.calculate_free_energy() == pytest.approx(0.0, abs=1e-9)


def test_a_total_that_overflows_raises_no_warning_anywhere() -> None:
    """The result was right but a warnings-as-errors run failed on the ``sum``."""
    import warnings

    huge = np.full(2, 1e308)  # finite entries whose total is infinite
    even = np.full(2, 0.5)
    matrix = np.full((2, 2), 1e308)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert normalize_mass(huge) == pytest.approx(even)
        assert _dispatcher()._normalize(huge, 2) == pytest.approx(even)
        assert HomeostaticInference._normalize(huge, 2) == pytest.approx(even)
        assert _normalize(huge) == pytest.approx(even)
        assert _as_probability_vector(huge, name="v") == pytest.approx(even)
        assert pomdp_normalize(huge) == pytest.approx(even)
        assert MatrixOps.normalize_columns(matrix)[:, 1] == pytest.approx(even)
        assert ensure_matrix_properties(matrix, "column_stochastic")[:, 0] == pytest.approx(even)
        assert compute_entropy(huge) == pytest.approx(np.log(2))
        assert kl_divergence(huge, huge) == 0.0
