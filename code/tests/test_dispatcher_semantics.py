"""Oracles for the dispatcher parameters that change what the agent does.

Every assertion here was written because a mutation of the corresponding line
survived the suite: `learning_rate` could be applied to the wrong side of the
blend, `temperature` could be dropped entirely, the precision clip could be
moved by three orders of magnitude, and the policy aggregation could be
inverted from a minimum to a maximum -- all without a single test failing.
Coverage reported those lines as executed the whole time.
"""

from __future__ import annotations

import numpy as np
import pytest
from cognitive.models.active_inference.base import ModelState
from cognitive.models.active_inference.dispatcher import (
    ActiveInferenceDispatcher,
    InferenceConfig,
    PolicyAggregation,
)
from cognitive.models.active_inference.generative_model import DiscreteGenerativeModel


def model() -> DiscreteGenerativeModel:
    return DiscreteGenerativeModel(
        A=np.array([[0.9, 0.1], [0.1, 0.9]]),
        B=np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2),
        C=np.array([0.0, 2.0]),
        D=np.array([0.7, 0.3]),
        E=np.array([0.5, 0.5]),
    )


def dispatcher(**overrides: object) -> ActiveInferenceDispatcher:
    settings: dict[str, object] = {
        "method": "variational",
        "policy_type": "discrete",
        "temporal_horizon": 3,
        "learning_rate": 0.25,
        "precision_init": 1.0,
        "seed": 0,
    }
    settings.update(overrides)
    return ActiveInferenceDispatcher(InferenceConfig(**settings), model())


def state(generative: DiscreteGenerativeModel) -> ModelState:
    return ModelState(generative.D.copy(), generative.E.copy(), 1.0, 0.0, 0.0)


@pytest.mark.parametrize("learning_rate", [0.05, 0.25, 0.5, 0.9, 1.0])
def test_variational_update_blends_toward_the_posterior_by_learning_rate(
    learning_rate: float,
) -> None:
    """The blend weights the posterior, not the prior, by ``learning_rate``.

    A rate of 0.5 is symmetric, so it alone cannot tell the two apart.
    """
    engine = dispatcher(learning_rate=learning_rate)
    current = state(engine.model)
    posterior = engine.model.posterior(1, current.beliefs)
    expected = (1.0 - learning_rate) * current.beliefs + learning_rate * posterior
    expected = expected / expected.sum()
    assert engine.dispatch_belief_update(1, current) == pytest.approx(expected, abs=1e-12)


def test_a_smaller_learning_rate_moves_beliefs_less() -> None:
    """Monotonicity in the parameter, independent of its exact algebra."""
    current = state(model())
    prior = current.beliefs.copy()
    distances = []
    for rate in (0.1, 0.5, 1.0):
        updated = dispatcher(learning_rate=rate).dispatch_belief_update(1, current.copy())
        distances.append(float(np.abs(updated - prior).sum()))
    assert distances[0] < distances[1] < distances[2]


@pytest.mark.parametrize("temperature", [0.25, 1.0, 4.0])
def test_policy_inference_applies_the_configured_temperature(temperature: float) -> None:
    """Policies are the softmax of negated energies scaled by temperature."""
    engine = dispatcher(temperature=temperature)
    current = state(engine.model)
    energies = engine.calculate_expected_free_energy(current)
    scaled = -energies / temperature
    expected = np.exp(scaled - scaled.max())
    expected = expected / expected.sum()
    assert engine.dispatch_policy_inference(current) == pytest.approx(expected, abs=1e-12)


def test_a_lower_temperature_sharpens_the_policy_distribution() -> None:
    """Temperature controls determinism, so the ordering of entropies is fixed."""
    current = state(model())
    entropies = []
    for temperature in (0.2, 1.0, 5.0):
        policies = dispatcher(temperature=temperature).dispatch_policy_inference(current.copy())
        entropies.append(float(-np.sum(policies * np.log(np.maximum(policies, 1e-300)))))
    assert entropies[0] < entropies[1] < entropies[2]


def test_minimum_aggregation_scores_each_action_by_its_best_continuation() -> None:
    """``min`` is optimistic: the action inherits its most favourable policy."""
    engine = dispatcher(policy_aggregation=PolicyAggregation.MIN)
    current = state(engine.model)
    values = engine.calculate_expected_free_energy(current)
    policies = engine.model.enumerate_policies(3, engine.config.policy_limit)
    energies = [
        engine.model.evaluate_policy(current.beliefs, policy, engine.config.discount_factor)
        for policy in policies
    ]
    for action in range(engine.model.num_actions):
        best = min(
            energy for policy, energy in zip(policies, energies, strict=True) if policy[0] == action
        )
        assert values[action] == pytest.approx(best, abs=1e-12)
        # The optimistic score can never exceed the average over continuations.
        average = np.mean(
            [
                energy
                for policy, energy in zip(policies, energies, strict=True)
                if policy[0] == action
            ]
        )
        assert values[action] <= average + 1e-12


def test_marginal_aggregation_reproduces_the_bayesian_model_average() -> None:
    """``marginal`` sums the policy posterior over policies sharing a first action."""
    temperature = 0.7
    engine = dispatcher(policy_aggregation=PolicyAggregation.MARGINAL, temperature=temperature)
    current = state(engine.model)
    policies = engine.model.enumerate_policies(3, engine.config.policy_limit)
    energies = np.array(
        [
            engine.model.evaluate_policy(current.beliefs, policy, engine.config.discount_factor)
            for policy in policies
        ]
    )
    weights = np.exp(-(energies - energies.min()) / temperature)
    expected = np.array(
        [
            weights[[index for index, policy in enumerate(policies) if policy[0] == action]].sum()
            for action in range(engine.model.num_actions)
        ]
    )
    expected = expected / expected.sum()
    assert engine.dispatch_policy_inference(current) == pytest.approx(expected, abs=1e-12)


def test_the_two_aggregations_disagree_and_are_both_selectable() -> None:
    """The setting must change the answer, or it is not a setting."""
    current = state(model())
    minimum = dispatcher(policy_aggregation="min").calculate_expected_free_energy(current.copy())
    marginal = dispatcher(policy_aggregation="marginal").calculate_expected_free_energy(
        current.copy()
    )
    # A soft minimum is never above the hard minimum it softens.
    assert np.all(marginal <= minimum + 1e-12)
    assert not np.allclose(marginal, minimum)
    with pytest.raises(ValueError):
        dispatcher(policy_aggregation="median")


def test_precision_update_is_clipped_at_its_documented_bounds() -> None:
    """Both ends of the clip are exercised, not just the interior."""
    engine = dispatcher()
    # A vanishing prediction error drives the target to 1e8, so repeated
    # updates must converge to the upper bound and stop there.
    for _ in range(500):
        precision = engine.update_precision(0.0)
    assert precision == pytest.approx(1e6)
    assert engine.update_precision(0.0) == pytest.approx(1e6)

    # A huge error drives the target to zero, and the lower bound holds.
    engine.config.precision_init = 1.0
    for _ in range(500):
        precision = engine.update_precision(1e12)
    assert precision == pytest.approx(0.1)

    with pytest.raises(ValueError, match="prediction_error"):
        engine.update_precision(-1.0)
    with pytest.raises(ValueError, match="prediction_error"):
        engine.update_precision(float("nan"))


def test_goal_prior_enters_the_policy_posterior_exactly_once() -> None:
    """A goal prior shifts energies by its log, and mean-field does not double it."""
    engine = dispatcher()
    current = state(engine.model)
    prior = np.array([0.8, 0.2])
    without = engine.calculate_expected_free_energy(current)
    with_prior = engine.calculate_expected_free_energy(current, prior)
    assert with_prior == pytest.approx(without - np.log(prior + 1e-12), abs=1e-9)

    mean_field = dispatcher(method="mean_field")
    variational = dispatcher(method="variational")
    assert mean_field.dispatch_policy_inference(current.copy(), prior) == pytest.approx(
        variational.dispatch_policy_inference(current.copy(), prior), abs=1e-9
    )
    with pytest.raises(ValueError, match="goal_prior must have shape"):
        engine.calculate_expected_free_energy(current, np.array([0.5, 0.3, 0.2]))


def test_the_default_aggregation_is_the_bayesian_model_average() -> None:
    """Version 2.0.0 made ``marginal`` the default; ``min`` stays selectable."""
    current = state(model())
    default = dispatcher().calculate_expected_free_energy(current.copy())
    marginal = dispatcher(policy_aggregation="marginal").calculate_expected_free_energy(
        current.copy()
    )
    minimum = dispatcher(policy_aggregation="min").calculate_expected_free_energy(current.copy())
    assert default == pytest.approx(marginal, abs=1e-12)
    assert not np.allclose(default, minimum)
