"""Regression tests for four behaviour changes, each reproduced before it was fixed.

* ``AdaptiveControl.compute_policy_prior`` ignored ``prediction_error``: the
  exploration term was the same constant for every entry, so the max-shift that
  precedes ``exp`` removed it.
* ``SimplePOMDP`` accepted ``inference.seed``, ``temperature``, and horizon keys
  and read none of them.
* ``ContinuousActiveInference`` precision learning chased ``1 / residual**2``
  without bound: one zero-residual step set the precision to ``1e12``.
* ``goal_prior`` entered the variational and sampling policies as
  ``ln(prior) / T`` but the mean-field policy as ``ln(prior)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cognitive.models.active_inference.base import ModelState
from cognitive.models.active_inference.dispatcher import ActiveInferenceDispatcher, InferenceConfig
from cognitive.models.active_inference.generative_model import DiscreteGenerativeModel
from cognitive.models.active_inference.homeostatic import AdaptiveControl
from Things.Continuous_Generic.continuous_generic import ContinuousActiveInference
from Things.Simple_POMDP.simple_pomdp import SimplePOMDP, compute_expected_free_energy

# --- AdaptiveControl ---------------------------------------------------------


def _state(prediction_error: float) -> ModelState:
    return ModelState(np.array([0.7, 0.2, 0.1]), np.full(2, 0.5), 1.0, 0.0, prediction_error)


GOAL = np.array([0.2, 0.5, 0.3])


def test_adaptive_prior_responds_to_prediction_error() -> None:
    control = AdaptiveControl(exploration_weight=0.3)
    calm = control.compute_policy_prior(_state(0.0), GOAL)
    surprised = control.compute_policy_prior(_state(4.0), GOAL)
    assert not np.allclose(calm, surprised)
    # More surprise flattens the prior toward uniform: exploration.
    assert np.ptp(surprised) < np.ptp(calm)


def test_adaptive_prior_matches_the_documented_scale() -> None:
    weight, error = 0.3, 2.0
    control = AdaptiveControl(exploration_weight=weight)
    score = -np.abs(_state(error).beliefs - GOAL)
    logits = (1 - weight) / (1 + weight * error) * score
    expected = np.exp(logits - logits.max())
    assert control.compute_policy_prior(_state(error), GOAL) == pytest.approx(expected)


def test_adaptive_prior_without_exploration_weight_ignores_prediction_error() -> None:
    """With ``exploration_weight=0`` the error is gated out, by design."""
    control = AdaptiveControl(exploration_weight=0.0)
    assert control.compute_policy_prior(_state(0.0), GOAL) == pytest.approx(
        control.compute_policy_prior(_state(9.0), GOAL)
    )


def test_adaptive_prior_is_monotone_in_prediction_error() -> None:
    control = AdaptiveControl(exploration_weight=0.5)
    spreads = [np.ptp(control.compute_policy_prior(_state(e), GOAL)) for e in (0.0, 0.5, 2.0, 8.0)]
    assert spreads == sorted(spreads, reverse=True)
    assert spreads[0] > spreads[-1]


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_adaptive_prior_rejects_an_invalid_prediction_error(bad: float) -> None:
    with pytest.raises(ValueError, match="prediction_error"):
        AdaptiveControl().compute_policy_prior(_state(bad), GOAL)


# --- SimplePOMDP -------------------------------------------------------------


def _pomdp_config(output_dir: Path, **inference: Any) -> dict[str, Any]:
    """Return a two-state configuration whose action prior favours action 0."""
    return {
        "model": {"name": "suspect"},
        "state_space": {"num_states": 2},
        "observation_space": {"num_observations": 2},
        "action_space": {"num_actions": 2},
        "matrices": {
            "A_matrix": {
                "shape": [2, 2],
                "initialization": "identity_based",
                "initialization_params": {"strength": 0.7},
                "constraints": ["column_stochastic"],
            },
            "B_matrix": {
                "shape": [2, 2, 2],
                "initialization": "identity_based",
                "initialization_params": {"strength": 0.6},
                "constraints": ["column_stochastic"],
            },
            "C_matrix": {
                "shape": [2],
                "initialization": "log_preferences",
                "initialization_params": {"preferences": [1.0, 0.0]},
            },
            "D_matrix": {"shape": [2], "initialization": "uniform"},
            "E_matrix": {
                "shape": [2],
                "initialization": "log_preferences",
                "initialization_params": {"preferences": [0.7, 0.3]},
            },
        },
        "inference": {"learning_rate": 1.0, **inference},
        "visualization": {"output_dir": str(output_dir)},
    }


def _observations(model: SimplePOMDP, steps: int = 40) -> list[int]:
    return [model.step(0)[0] for _ in range(steps)]


def test_inference_seed_controls_the_random_stream(tmp_path: Path) -> None:
    first = _observations(SimplePOMDP(_pomdp_config(tmp_path, seed=1)))
    again = _observations(SimplePOMDP(_pomdp_config(tmp_path, seed=1)))
    other = _observations(SimplePOMDP(_pomdp_config(tmp_path, seed=2)))
    assert first == again
    assert first != other


def test_root_and_inference_seed_must_agree(tmp_path: Path) -> None:
    config = _pomdp_config(tmp_path, seed=1)
    config["seed"] = 2
    with pytest.raises(ValueError, match="disagree"):
        SimplePOMDP(config)
    config["seed"] = 1
    agreed = _observations(SimplePOMDP(config))
    assert agreed == _observations(SimplePOMDP(_pomdp_config(tmp_path, seed=1)))


@pytest.mark.parametrize("seed", [-1, 1.5, "7"])
def test_seed_must_be_a_non_negative_integer(tmp_path: Path, seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        SimplePOMDP(_pomdp_config(tmp_path, seed=seed))


def _action_distribution(model: SimplePOMDP, draws: int) -> np.ndarray:
    counts = np.bincount([model._select_action()[0] for _ in range(draws)], minlength=2)
    return counts / draws


def test_temperature_sets_the_action_selection_softmax(tmp_path: Path) -> None:
    temperature = 0.8
    model = SimplePOMDP(_pomdp_config(tmp_path, temperature=temperature, seed=3))
    totals = np.array(
        [
            compute_expected_free_energy(model.A, model.B, model.C, model.state.beliefs, a)[0]
            for a in range(2)
        ]
    )
    scores = totals - np.log(model.E)
    expected = np.exp(-scores / temperature)
    expected /= expected.sum()
    assert 0.1 < expected.min() < 0.9  # an interior target, so the check has power
    measured = _action_distribution(model, draws=4000)
    assert measured == pytest.approx(expected, abs=0.03)


def test_low_temperature_is_greedy_and_high_temperature_explores(tmp_path: Path) -> None:
    greedy = SimplePOMDP(_pomdp_config(tmp_path, temperature=1e-3, seed=4))
    assert _action_distribution(greedy, draws=200) == pytest.approx([1.0, 0.0])  # E favours 0
    hot = SimplePOMDP(_pomdp_config(tmp_path, temperature=1e6, seed=4))
    assert _action_distribution(hot, draws=4000) == pytest.approx([0.5, 0.5], abs=0.03)


def test_without_temperature_selection_is_the_greedy_limit(tmp_path: Path) -> None:
    model = SimplePOMDP(_pomdp_config(tmp_path, seed=5))
    chosen = {model._select_action()[0] for _ in range(50)}
    assert len(chosen) == 1


# An explicit null is not a temperature; reading it as "absent" would hide a typo.
@pytest.mark.parametrize("temperature", [0, -1.0, float("inf"), float("nan"), "hot", True, None])
def test_invalid_temperature_is_rejected(tmp_path: Path, temperature: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        SimplePOMDP(_pomdp_config(tmp_path, temperature=temperature))


@pytest.mark.parametrize("key", ["time_horizon", "temporal_horizon"])
def test_horizon_beyond_one_step_is_rejected(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError, match="must be 1"):
        SimplePOMDP(_pomdp_config(tmp_path, **{key: 5}))
    with pytest.raises(ValueError, match="integer"):
        SimplePOMDP(_pomdp_config(tmp_path, **{key: 1.5}))
    SimplePOMDP(_pomdp_config(tmp_path, **{key: 1}))


@pytest.mark.parametrize("key", ["num_iterations", "policy_learning_rate"])
def test_inference_fields_nothing_reads_are_rejected(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError, match="not used by SimplePOMDP"):
        SimplePOMDP(_pomdp_config(tmp_path, **{key: 3}))


# --- ContinuousActiveInference -----------------------------------------------


def _learner(**overrides: Any) -> ContinuousActiveInference:
    settings: dict[str, Any] = {
        "n_states": 1,
        "n_obs": 1,
        "n_orders": 1,
        "precision_learning_rate": 1.0,
    }
    settings.update(overrides)
    return ContinuousActiveInference(**settings)


def test_a_zero_residual_does_not_send_precision_to_the_floor_constant() -> None:
    agent = _learner()
    agent.step(np.array([0.0]))  # prediction matches: residual 0, target 1 / EPS
    assert agent.observation_precision[0] <= 1e3
    action, _ = agent.step(np.array([1.0]))
    # With precision 1e12 the next gradient step threw the belief to ~1e8.
    assert abs(agent.state.belief_means[0, 0]) <= 1.0
    assert abs(action[0]) <= 1e3


def test_a_huge_residual_does_not_freeze_the_update() -> None:
    agent = _learner(observation_precision=1.0)
    agent.step(np.array([1e6]))  # target 1e-12
    assert agent.observation_precision[0] >= agent.precision_min


def test_precision_stays_inside_its_bounds_over_a_long_run() -> None:
    agent = _learner(precision_min=0.5, precision_max=20.0, observation_precision=2.0)
    rng = np.random.default_rng(0)
    for scale in rng.choice([0.0, 1e-9, 1e-3, 1.0, 1e3, 1e6], size=300):
        agent.step(np.array([float(scale)]))
        assert 0.5 <= agent.observation_precision[0] <= 20.0
        assert np.all(np.isfinite(agent.state.belief_means))


def test_the_learned_target_is_clipped_to_the_configured_bounds() -> None:
    agent = _learner(precision_min=0.5, precision_max=4.0, observation_precision=1.0)
    agent.step(np.array([0.0]))  # target 1e12 -> clipped to 4.0, learning rate 1
    assert agent.observation_precision[0] == pytest.approx(4.0)
    agent.step(np.array([1e6]))  # target 1e-12 -> clipped to 0.5
    assert agent.observation_precision[0] == pytest.approx(0.5)


def test_the_learned_target_is_clipped_before_it_is_blended() -> None:
    """A learning rate below one moves part of the way toward the clipped target."""
    agent = _learner(
        precision_learning_rate=0.5,
        precision_min=0.5,
        precision_max=4.0,
        observation_precision=1.0,
    )
    agent.step(np.array([0.0]))  # target 1e12 -> 4.0; blend 0.5 * 1.0 + 0.5 * 4.0
    assert agent.observation_precision[0] == pytest.approx(2.5)


def test_precision_bounds_are_validated() -> None:
    for low, high in [(0.0, 1.0), (-1.0, 1.0), (2.0, 1.0), (1.0, 1.0)]:
        with pytest.raises(ValueError, match="precision bounds"):
            ContinuousActiveInference(precision_min=low, precision_max=high)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            ContinuousActiveInference(precision_max=bad)


def test_initial_precision_must_sit_inside_the_bounds_only_when_learning() -> None:
    with pytest.raises(ValueError, match="observation_precision must lie in"):
        ContinuousActiveInference(
            precision_learning_rate=0.1, observation_precision=5e3, precision_max=1e3
        )
    fixed = ContinuousActiveInference(
        precision_learning_rate=0.0, observation_precision=5e3, precision_max=1e3
    )
    assert fixed.observation_precision == pytest.approx([5e3, 5e3])


# --- goal_prior scaling ------------------------------------------------------


def _policy_model() -> DiscreteGenerativeModel:
    return DiscreteGenerativeModel(
        A=np.array([[0.9, 0.1], [0.1, 0.9]]),
        B=np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2),
        C=np.array([0.0, 2.0]),
        D=np.array([0.7, 0.3]),
        E=np.array([0.5, 0.5]),
    )


def _engine(method: str, temperature: float, **overrides: Any) -> ActiveInferenceDispatcher:
    return ActiveInferenceDispatcher(
        InferenceConfig(
            method, "discrete", 2, 0.5, 1.0, temperature=temperature, seed=11, **overrides
        ),
        _policy_model(),
    )


def _boltzmann_with_prior(temperature: float, prior: np.ndarray) -> np.ndarray:
    """``q(a) = E(a) exp(-G(a) / T) / Z`` from the prior-free energies."""
    model = _policy_model()
    state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
    energies = _engine("variational", temperature).calculate_expected_free_energy(state)
    weights = prior * np.exp(-(energies - energies.min()) / temperature)
    return weights / weights.sum()


@pytest.mark.parametrize("temperature", [0.25, 0.5, 2.0, 5.0])
@pytest.mark.parametrize("method", ["variational", "mean_field"])
def test_goal_prior_multiplies_the_tempered_likelihood(method: str, temperature: float) -> None:
    model = _policy_model()
    state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
    prior = np.array([0.85, 0.15])
    inferred = _engine(method, temperature).dispatch_policy_inference(state, prior)
    assert inferred == pytest.approx(_boltzmann_with_prior(temperature, prior), abs=1e-9)


@pytest.mark.parametrize("temperature", [0.25, 2.0])
def test_sampling_applies_the_goal_prior_like_the_exact_methods(temperature: float) -> None:
    model = _policy_model()
    state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
    prior = np.array([0.85, 0.15])
    inferred = _engine("sampling", temperature, num_samples=40000).dispatch_policy_inference(
        state, prior
    )
    assert inferred == pytest.approx(_boltzmann_with_prior(temperature, prior), abs=0.01)


@pytest.mark.parametrize("aggregation", ["min", "marginal"])
def test_methods_agree_on_a_goal_prior_at_any_temperature(aggregation: str) -> None:
    model = _policy_model()
    state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
    prior = np.array([0.6, 0.4])
    results = [
        _engine(method, 3.0, policy_aggregation=aggregation).dispatch_policy_inference(state, prior)
        for method in ("variational", "mean_field")
    ]
    assert results[0] == pytest.approx(results[1], abs=1e-9)


def test_goal_prior_shifts_the_energy_by_temperature_times_its_log() -> None:
    temperature = 2.5
    engine = _engine("variational", temperature)
    model = _policy_model()
    state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
    prior = np.array([0.8, 0.2])
    shifted = engine.calculate_expected_free_energy(state, prior)
    plain = engine.calculate_expected_free_energy(state)
    assert shifted == pytest.approx(plain - temperature * np.log(prior + 1e-12), abs=1e-9)
