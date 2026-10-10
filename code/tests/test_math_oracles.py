"""Closed-form and identity oracles for the package's Active Inference math.

Every assertion here compares an implementation against a quantity derived
independently -- an analytic value, an algebraically equivalent decomposition,
or a finite-difference gradient -- rather than against the implementation's own
output.  Finiteness and normalization checks live in ``test_truth_audit.py``;
this module exists to catch a result that is well-formed and wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from cognitive.models.active_inference.generative_model import DiscreteGenerativeModel
from cognitive.models.matrices.matrix_ops import MatrixOps
from cognitive.utils.matrix_utils import compute_entropy
from cognitive.utils.matrix_utils import expected_free_energy as efe_from_matrices
from Things.Continuous_Generic import ContinuousActiveInference


def random_model(rng: np.random.Generator, shape: tuple[int, int, int]) -> DiscreteGenerativeModel:
    """Build a valid random generative model with the requested |O|, |S|, |A|."""
    observations, states, actions = shape
    likelihood = rng.random((observations, states)) + 1e-3
    likelihood /= likelihood.sum(axis=0)
    transitions = rng.random((states, states, actions)) + 1e-3
    transitions /= transitions.sum(axis=0)
    prior = rng.random(states)
    action_prior = rng.random(actions)
    return DiscreteGenerativeModel(
        A=likelihood,
        B=transitions,
        C=rng.normal(size=observations),
        D=prior / prior.sum(),
        E=action_prior / action_prior.sum(),
    )


SHAPES = [(2, 2, 2), (3, 2, 2), (2, 3, 4), (4, 3, 2), (5, 4, 3)]


def _entropy(distribution: np.ndarray) -> float:
    values = np.asarray(distribution, dtype=float)
    return float(-np.sum(np.where(values > 0, values * np.log(np.maximum(values, 1e-300)), 0.0)))


@pytest.mark.parametrize("shape", SHAPES)
def test_efe_decompositions_agree(shape: tuple[int, int, int]) -> None:
    """``risk + ambiguity`` equals ``expected surprise - information gain``.

    These are the two standard decompositions of expected free energy and they
    are algebraically identical.  Subtracting the epistemic gain from
    ``risk + ambiguity`` -- as this package did before 1.1.0 -- breaks the
    identity by counting information seeking twice.
    """
    model = random_model(np.random.default_rng(sum(shape)), shape)
    for action in range(model.num_actions):
        total, risk, ambiguity, epistemic = model.expected_free_energy(model.D, action)
        predicted_states = model.predict_states(model.D, action)
        predicted_observations = model.predict_observations(predicted_states)
        expected_surprise = -float(
            np.sum(predicted_observations * np.log(np.maximum(model.preferences, 1e-300)))
        )
        assert total == pytest.approx(risk + ambiguity, abs=1e-12)
        assert total == pytest.approx(expected_surprise - epistemic, abs=1e-9)


@pytest.mark.parametrize("shape", SHAPES)
def test_efe_components_match_their_definitions(shape: tuple[int, int, int]) -> None:
    """Each reported component equals its textbook definition."""
    model = random_model(np.random.default_rng(7 + sum(shape)), shape)
    for action in range(model.num_actions):
        _total, risk, ambiguity, epistemic = model.expected_free_energy(model.D, action)
        predicted_states = model.predict_states(model.D, action)
        predicted_observations = model.predict_observations(predicted_states)

        expected_risk = float(
            np.sum(
                predicted_observations
                * np.log(
                    np.maximum(predicted_observations, 1e-300)
                    / np.maximum(model.preferences, 1e-300)
                )
            )
        )
        expected_ambiguity = float(
            np.sum(predicted_states * np.array([_entropy(model.A[:, s]) for s in range(shape[1])]))
        )
        posterior_entropy = float(
            np.sum(
                [
                    predicted_observations[o] * _entropy(model.posterior(o, predicted_states))
                    for o in range(shape[0])
                ]
            )
        )
        assert risk == pytest.approx(expected_risk, abs=1e-9)
        assert ambiguity == pytest.approx(expected_ambiguity, abs=1e-9)
        # Ambiguity is H(o | s); the epistemic gain is the mutual information
        # H(o) - H(o | s). Both must be recoverable from the same joint.
        assert epistemic == pytest.approx(_entropy(predicted_states) - posterior_entropy, abs=1e-9)
        assert ambiguity + epistemic == pytest.approx(_entropy(predicted_observations), abs=1e-9)


@pytest.mark.parametrize("shape", SHAPES)
def test_matrix_utils_efe_matches_generative_model(shape: tuple[int, int, int]) -> None:
    """The standalone helper and the model class compute the same objective."""
    model = random_model(np.random.default_rng(100 + sum(shape)), shape)
    for action in range(model.num_actions):
        assert efe_from_matrices(model.A, model.B, model.C, model.D, action) == pytest.approx(
            model.expected_free_energy(model.D, action)[0], abs=1e-9
        )


def test_ambiguity_is_zero_for_a_noiseless_likelihood() -> None:
    """A deterministic ``A`` carries no observation uncertainty."""
    model = DiscreteGenerativeModel(
        A=np.eye(3),
        B=np.stack([np.eye(3)], axis=2),
        C=np.array([0.0, 1.0, 0.0]),
        D=np.array([0.2, 0.5, 0.3]),
        E=np.array([1.0]),
    )
    _total, _risk, ambiguity, epistemic = model.expected_free_energy(model.D, 0)
    assert ambiguity == pytest.approx(0.0, abs=1e-12)
    # With a noiseless likelihood every observation identifies the state, so
    # the information gain is the full state entropy.
    assert epistemic == pytest.approx(_entropy(model.D), abs=1e-9)


def test_epistemic_gain_is_zero_for_an_uninformative_likelihood() -> None:
    """An ``A`` whose columns are identical carries no information about ``s``."""
    model = DiscreteGenerativeModel(
        A=np.full((2, 3), 0.5),
        B=np.stack([np.eye(3)], axis=2),
        C=np.array([0.0, 0.0]),
        D=np.array([0.2, 0.5, 0.3]),
        E=np.array([1.0]),
    )
    _total, _risk, ambiguity, epistemic = model.expected_free_energy(model.D, 0)
    assert epistemic == pytest.approx(0.0, abs=1e-12)
    assert ambiguity == pytest.approx(np.log(2.0), abs=1e-9)


def test_risk_is_zero_when_predictions_match_preferences() -> None:
    """Risk is a divergence, so it vanishes exactly at the preferred outcome."""
    likelihood = np.array([[0.8, 0.3], [0.2, 0.7]])
    beliefs = np.array([0.4, 0.6])
    predicted = likelihood @ beliefs
    model = DiscreteGenerativeModel(
        A=likelihood,
        B=np.stack([np.eye(2)], axis=2),
        C=np.log(predicted),
        D=beliefs,
        E=np.array([1.0]),
    )
    _total, risk, _ambiguity, _epistemic = model.expected_free_energy(beliefs, 0)
    assert risk == pytest.approx(0.0, abs=1e-12)


def test_preferred_action_has_the_lower_expected_free_energy() -> None:
    """A behavioural oracle: the action that reaches the preferred observation wins."""
    stay = np.eye(2)
    switch = np.array([[0.0, 1.0], [1.0, 0.0]])
    model = DiscreteGenerativeModel(
        A=np.array([[0.95, 0.05], [0.05, 0.95]]),
        B=np.stack([stay, switch], axis=2),
        C=np.array([0.0, 4.0]),
        D=np.array([1.0, 0.0]),
        E=np.array([0.5, 0.5]),
    )
    staying = model.expected_free_energy(model.D, 0)[0]
    switching = model.expected_free_energy(model.D, 1)[0]
    assert switching < staying


def test_epistemic_weight_reproduces_the_pre_1_1_objective() -> None:
    """``epistemic_weight=1`` recovers the objective used before the correction."""
    model = random_model(np.random.default_rng(3), (3, 3, 2))
    for action in range(model.num_actions):
        total, _risk, _ambiguity, epistemic = model.expected_free_energy(model.D, action)
        weighted = model.evaluate_policy(model.D, (action,), 1.0, epistemic_weight=1.0)
        assert weighted == pytest.approx(total - epistemic, abs=1e-12)
        assert model.evaluate_policy(model.D, (action,), 1.0) == pytest.approx(total, abs=1e-12)
    with pytest.raises(ValueError):
        model.evaluate_policy(model.D, (0,), 1.0, epistemic_weight=-0.1)


def test_policy_evaluation_applies_the_discount_factor() -> None:
    """A two-step policy is the discounted sum of its per-step energies."""
    model = random_model(np.random.default_rng(5), (2, 2, 2))
    policy = (0, 1)
    discount = 0.5
    first = model.expected_free_energy(model.D, policy[0])[0]
    second = model.expected_free_energy(model.predict_states(model.D, policy[0]), policy[1])[0]
    assert model.evaluate_policy(model.D, policy, discount) == pytest.approx(
        first + discount * second, abs=1e-12
    )


def test_posterior_matches_bayes_rule() -> None:
    """The state posterior is exactly the normalized likelihood-times-prior."""
    model = random_model(np.random.default_rng(9), (4, 3, 2))
    prior = model.D
    for observation in range(model.num_observations):
        unnormalized = model.A[observation] * prior
        assert model.posterior(observation, prior) == pytest.approx(
            unnormalized / unnormalized.sum(), abs=1e-12
        )


def test_enumerate_policies_is_ordered_and_truncates_lazily() -> None:
    """Truncation keeps the lexicographic prefix rather than an arbitrary subset."""
    model = random_model(np.random.default_rng(1), (2, 2, 3))
    full = model.enumerate_policies(2)
    assert full == sorted(full)
    assert len(full) == 3**2
    assert model.enumerate_policies(2, limit=4) == full[:4]
    # A limit far above the policy count returns the whole space, and a large
    # horizon with a small limit must not materialize the full product.
    assert model.enumerate_policies(2, limit=10_000) == full
    assert len(model.enumerate_policies(20, limit=5)) == 5


def test_entropy_and_divergence_closed_forms() -> None:
    """Analytic values, including the exact boundaries of each definition."""
    assert MatrixOps.compute_entropy(np.array([0.5, 0.5])) == pytest.approx(np.log(2.0))
    assert MatrixOps.compute_entropy(np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert MatrixOps.compute_kl_divergence(
        np.array([0.5, 0.5]), np.array([0.5, 0.5])
    ) == pytest.approx(0.0)
    # D_KL([1, 0] || [0.5, 0.5]) = log 2 exactly.
    assert MatrixOps.compute_kl_divergence(
        np.array([1.0, 0.0]), np.array([0.5, 0.5])
    ) == pytest.approx(np.log(2.0))


def test_compute_entropy_normalizes_along_the_axis_it_sums() -> None:
    """Row entropies come from row-normalized values, not column-normalized ones."""
    matrix = np.array([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]])
    rows = compute_entropy(matrix, axis=-1)
    assert rows[0] == pytest.approx(-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)))
    assert rows[1] == pytest.approx(np.log(2.0))
    columns = compute_entropy(matrix, axis=0)
    first = matrix[:, 0] / matrix[:, 0].sum()
    assert columns[0] == pytest.approx(_entropy(first))


@pytest.mark.parametrize(
    "dynamics",
    [
        np.eye(2),
        np.array([[0.0, 1.0], [-1.0, 0.3]]),
        np.array([[0.5, -0.2], [0.7, 1.1]]),
    ],
)
def test_step_follows_free_energy_gradient(dynamics: np.ndarray) -> None:
    """The applied belief update is exact gradient descent on the reported energy.

    Omitting the dynamics Jacobian transpose from the update passes for the
    identity matrix and fails for every other dynamics matrix, so the identity
    case alone cannot detect it.
    """
    rng = np.random.default_rng(0)
    agent = ContinuousActiveInference(
        n_states=2, n_obs=2, n_orders=3, dt=0.01, alpha=0.2, dynamics_matrix=dynamics
    )
    agent.state.belief_means = rng.normal(size=(2, 3))
    observation = rng.normal(size=2)
    start = agent.state.belief_means.copy()

    step = 1e-6
    gradient = np.zeros_like(start)
    for row in range(start.shape[0]):
        for column in range(start.shape[1]):
            forward = start.copy()
            forward[row, column] += step
            backward = start.copy()
            backward[row, column] -= step
            gradient[row, column] = (
                agent._compute_free_energy(observation, forward)
                - agent._compute_free_energy(observation, backward)
            ) / (2 * step)

    agent.step(observation)
    applied = (agent.state.belief_means - start) / agent.dt
    assert applied == pytest.approx(-agent.alpha * gradient, abs=1e-6)


def test_continuous_step_reduces_free_energy_for_a_small_step() -> None:
    """Gradient descent with a small step must not increase the objective."""
    agent = ContinuousActiveInference(
        n_states=2,
        n_obs=2,
        n_orders=3,
        dt=0.001,
        alpha=0.05,
        dynamics_matrix=np.array([[0.0, 1.0], [-1.0, 0.2]]),
    )
    observation = np.array([0.7, -0.4])
    before = agent._compute_free_energy(observation, agent.state.belief_means.copy())
    agent.step(observation)
    after = agent._compute_free_energy(observation, agent.state.belief_means.copy())
    assert after <= before + 1e-12


def test_dynamics_weight_is_validated_and_scales_the_dynamic_term() -> None:
    """The dynamic term's weight appears in the reported energy exactly once."""
    means = np.array([[0.3, -0.2, 0.1], [0.4, 0.5, -0.3]])
    observation = np.array([0.1, 0.2])
    energies = {}
    for weight in (0.0, 1.0, 2.0):
        agent = ContinuousActiveInference(n_states=2, n_obs=2, n_orders=3, dynamics_weight=weight)
        agent.state.belief_means = means.copy()
        energies[weight] = agent._compute_free_energy(observation, means)
    dynamic = energies[1.0] - energies[0.0]
    assert energies[2.0] == pytest.approx(energies[0.0] + 2.0 * dynamic, abs=1e-12)
    with pytest.raises(ValueError):
        ContinuousActiveInference(dynamics_weight=-0.1)
