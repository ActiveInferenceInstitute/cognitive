import logging
from typing import Any

import numpy as np
import seaborn as sns

# Configure plotting style
sns.set_style("whitegrid")  # Using seaborn's style directly instead of plt.style


def log_test_case(test_name: str, test_data: dict[str, Any]) -> None:
    """Log test case information."""
    logging.info(f"Running test: {test_name}")
    for key, value in test_data.items():
        logging.debug(f"{key}: {value}")


def generate_test_data(size: int = 10) -> dict[str, Any]:
    """Generate test data for visualization tests."""
    rng = np.random.default_rng(0)
    timesteps = np.arange(size)
    internal_states = rng.random((size, 3))
    external_states = rng.random((size, 2))
    free_energies = rng.random(size)
    prior = rng.random((size, 3))
    posterior = rng.random((size, 3))
    observations = rng.random((size, 2))

    # Generate Markov blanket states
    sensory_states = rng.random((size, 2))
    internal_states_mb = rng.random((size, 3))
    active_states = rng.random((size, 2))

    return {
        "internal_states": internal_states,
        "external_states": external_states,
        "free_energies": free_energies,
        "prior": prior,
        "posterior": posterior,
        "observations": observations,
        "timesteps": timesteps,
        "sensory_states": sensory_states,
        "internal_states_mb": internal_states_mb,
        "active_states": active_states,
    }


# Export the functions
__all__ = ["log_test_case", "generate_test_data"]
