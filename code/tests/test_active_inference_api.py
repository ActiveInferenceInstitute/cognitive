import matplotlib.pyplot as plt
import numpy as np
import src
import yaml
from src.models import active_inference
from src.models.active_inference.base import ModelState
from src.models.active_inference.dispatcher import (
    ActiveInferenceDispatcher,
    InferenceConfig,
    InferenceMethod,
    PolicyType,
)
from src.models.active_inference.homeostatic import (
    HomeostaticControl,
    HomeostaticFactory,
    HomestaticControl,
)


def test_mean_field_dispatcher_updates_normalized_beliefs_and_policies():
    dispatcher = ActiveInferenceDispatcher(
        InferenceConfig(
            method=InferenceMethod.MEAN_FIELD,
            policy_type=PolicyType.DISCRETE,
            temporal_horizon=1,
            learning_rate=0.5,
            precision_init=1.0,
            temperature=1.0,
        )
    )
    state = ModelState(
        beliefs=np.array([0.55, 0.45]),
        policies=np.array([0.5, 0.5]),
        precision=1.0,
        free_energy=0.0,
        prediction_error=0.0,
    )

    beliefs = dispatcher.dispatch_belief_update(
        np.array([0.85, 0.15]),
        state,
        generative_matrix=np.eye(2),
    )
    policies = dispatcher.dispatch_policy_inference(
        state,
        goal_prior=np.array([0.8, 0.2]),
        exploration_weight=0.0,
    )

    assert np.all(np.isfinite(beliefs))
    assert np.isclose(np.sum(beliefs), 1.0)
    assert beliefs[0] > state.beliefs[0]
    assert np.all(np.isfinite(policies))
    assert np.isclose(np.sum(policies), 1.0)
    assert policies[0] > policies[1]


def test_sampling_policy_proposal_handles_zero_vector():
    dispatcher = ActiveInferenceDispatcher(
        InferenceConfig(
            method=InferenceMethod.SAMPLING,
            policy_type=PolicyType.DISCRETE,
            temporal_horizon=1,
            learning_rate=0.1,
            precision_init=1.0,
            num_samples=8,
        )
    )

    proposal = dispatcher._propose_policy(np.zeros(3))

    assert np.all(np.isfinite(proposal))
    assert np.isclose(np.sum(proposal), 1.0)
    assert np.all(proposal >= 0)


def test_homeostatic_factory_creates_step_capable_model(tmp_path):
    config_path = tmp_path / "homeostatic.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "state_spaces": {
                    "environment": {
                        "dimensions": [2],
                        "labels": {"states": ["LOW", "HIGH"]},
                        "mappings": {"identity": [[1.0, 0.0], [0.0, 1.0]]},
                    },
                    "observation": {
                        "dimensions": [2],
                        "labels": {"observations": ["low", "high"]},
                        "mappings": {"identity": [[1.0, 0.0], [0.0, 1.0]]},
                    },
                    "action": {
                        "dimensions": [2],
                        "labels": {"actions": ["stay", "switch"]},
                        "mappings": {"identity": [[1.0, 0.0], [0.0, 1.0]]},
                    },
                },
                "observation_model": {
                    "likelihood_matrix": [[0.9, 0.1], [0.1, 0.9]]
                },
                "transition_model": {
                    "transition_matrices": {
                        "stay": [[0.9, 0.1], [0.1, 0.9]],
                        "switch": [[0.1, 0.9], [0.9, 0.1]],
                    },
                    "temporal_horizon": 1,
                },
                "inference": {
                    "method": "variational",
                    "policy_type": "discrete",
                    "temporal_horizon": 1,
                    "learning_rate": 0.2,
                    "precision_init": 1.0,
                },
                "target_state": [0.8, 0.2],
                "initial_beliefs": [0.5, 0.5],
            }
        )
    )

    model = HomeostaticFactory.create_basic(config_path)
    observation, free_energy = model.step()
    fig = model.visualize("beliefs")

    assert HomestaticControl is HomeostaticControl
    assert model.state_space.validate()
    assert model.observation_space.validate()
    assert model.action_space.validate()
    assert isinstance(observation, int)
    assert np.isfinite(free_energy)
    assert np.isclose(np.sum(model.state.beliefs), 1.0)
    assert np.isclose(np.sum(model.state.policies), 1.0)

    plt.close(fig)


def test_active_inference_package_exports_public_api():
    assert src.ActiveInferenceDispatcher is active_inference.ActiveInferenceDispatcher
    assert src.HomeostaticFactory is active_inference.HomeostaticFactory
    assert active_inference.HomeostaticControl is HomeostaticControl
    assert active_inference.HomestaticControl is HomeostaticControl
