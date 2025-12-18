---
title: Active Inference Implementation Guide
type: implementation
status: stable
created: 2025-01-01
updated: 2025-01-01
tags:
  - active_inference
  - implementation
  - python
  - practical_guide
semantic_relations:
  - type: implements
    links:
      - [[mathematics/free_energy_principle]]
      - [[cognitive/active_inference]]
      - [[tools/src/models/|Implementation Tools]]
---

# Active Inference Implementation Guide

This document provides practical implementations of active inference principles, offering code examples, algorithmic details, and step-by-step guides for applying active inference to real-world problems. The implementations build on the mathematical foundations while providing actionable code for researchers and developers.

## 🐍 Python Implementation Framework

### Core Active Inference Components

#### Generative Model Definition
```python
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from typing import Dict, Tuple, List

class GenerativeModel:
    """Generative model for active inference agent."""

    def __init__(self, n_states: int, n_actions: int, n_observations: int):
        """Initialize generative model components.

        Args:
            n_states: Number of hidden states
            n_actions: Number of possible actions
            n_observations: Number of possible observations
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_observations = n_observations

        # Initialize model parameters
        self.A = self.initialize_A_matrix()  # Likelihood: p(o|s)
        self.B = self.initialize_B_matrix()  # Transition: p(s'|s,a)
        self.C = self.initialize_C_vector()  # Prior preference: p(o*)
        self.D = self.initialize_D_vector()  # Prior state: p(s)

    def initialize_A_matrix(self) -> jnp.ndarray:
        """Initialize likelihood matrix A."""
        # Start with uniform distribution, will be learned
        A = jnp.ones((self.n_observations, self.n_states)) / self.n_observations
        return A

    def initialize_B_matrix(self) -> jnp.ndarray:
        """Initialize transition matrix B."""
        # Initialize with identity (no action changes state)
        B = jnp.zeros((self.n_states, self.n_states, self.n_actions))
        for a in range(self.n_actions):
            B = B.at[:, :, a].set(jnp.eye(self.n_states))
        return B

    def initialize_C_vector(self) -> jnp.ndarray:
        """Initialize prior preference vector C."""
        # Uniform preferences initially
        C = jnp.ones(self.n_observations) / self.n_observations
        return C

    def initialize_D_vector(self) -> jnp.ndarray:
        """Initialize prior state vector D."""
        # Uniform prior over states
        D = jnp.ones(self.n_states) / self.n_states
        return D
```

#### Variational Free Energy Computation
```python
class VariationalInference:
    """Variational inference for active inference."""

    def __init__(self, model: GenerativeModel):
        self.model = model

        # Initialize variational parameters
        self.qs = self.model.D.copy()  # Variational state posterior
        self.qs_prev = self.qs.copy()

    def compute_variational_free_energy(self, observation: int) -> float:
        """Compute variational free energy for current belief state.

        Args:
            observation: Current observation index

        Returns:
            Variational free energy value
        """
        # Expected log likelihood: E_{q(s)}[ln p(o|s)]
        expected_log_likelihood = jnp.sum(
            self.qs * jnp.log(self.model.A[observation, :])
        )

        # Entropy of variational posterior: -E_{q(s)}[ln q(s)]
        entropy = -jnp.sum(self.qs * jnp.log(self.qs + 1e-16))

        # Expected log prior: E_{q(s)}[ln p(s)]
        expected_log_prior = jnp.sum(self.qs * jnp.log(self.model.D))

        # Variational free energy
        F = expected_log_prior - expected_log_likelihood - entropy

        return float(F)

    def update_beliefs(self, observation: int, learning_rate: float = 0.1):
        """Update variational beliefs using gradient descent on free energy.

        Args:
            observation: Current observation
            learning_rate: Learning rate for belief updates
        """
        def free_energy(qs):
            """Free energy as function of beliefs."""
            # Normalize beliefs
            qs_norm = qs / jnp.sum(qs)

            expected_log_likelihood = jnp.sum(
                qs_norm * jnp.log(self.model.A[observation, :] + 1e-16)
            )
            entropy = -jnp.sum(qs_norm * jnp.log(qs_norm + 1e-16))
            expected_log_prior = jnp.sum(qs_norm * jnp.log(self.model.D + 1e-16))

            return expected_log_prior - expected_log_likelihood - entropy

        # Gradient descent on free energy
        grad_fn = grad(free_energy)
        gradient = grad_fn(self.qs)

        # Update beliefs
        self.qs = self.qs - learning_rate * gradient
        self.qs = jnp.maximum(self.qs, 1e-16)  # Ensure positivity
        self.qs = self.qs / jnp.sum(self.qs)  # Normalize

    def infer_states(self, observation: int, n_iterations: int = 10) -> jnp.ndarray:
        """Perform variational inference to infer hidden states.

        Args:
            observation: Current observation
            n_iterations: Number of inference iterations

        Returns:
            Posterior belief over states
        """
        for _ in range(n_iterations):
            self.update_beliefs(observation)

        return self.qs
```

#### Expected Free Energy for Action Selection
```python
class ActionSelection:
    """Action selection using expected free energy minimization."""

    def __init__(self, model: GenerativeModel, inference: VariationalInference):
        self.model = model
        self.inference = inference

    def compute_expected_free_energy(self, action: int, qs: jnp.ndarray) -> float:
        """Compute expected free energy for a given action.

        Args:
            action: Action index
            qs: Current state beliefs

        Returns:
            Expected free energy value
        """
        G = 0.0

        # Sum over possible next states and observations
        for s_next in range(self.model.n_states):
            # Transition probability: p(s'|s,a)
            transition_prob = self.model.B[s_next, :, action]

            # Predictive posterior: q(s'|a) = ∑_s q(s) p(s'|s,a)
            qs_next = qs * transition_prob
            qs_next = qs_next / jnp.sum(qs_next)

            for o_next in range(self.model.n_observations):
                # Likelihood: p(o'|s')
                likelihood = self.model.A[o_next, s_next]

                # Posterior predictive: q(o'|a) = ∑_{s'} q(s'|a) p(o'|s')
                qo_next = jnp.sum(qs_next * self.model.A[o_next, :])

                # Expected free energy components
                extrinsic_term = qo_next * jnp.log(qo_next / (self.model.C[o_next] + 1e-16))
                epistemic_term = qs_next[s_next] * jnp.log(qs_next[s_next] / (qs[s_next] + 1e-16))

                G += likelihood * jnp.sum(qs * transition_prob) * (extrinsic_term + epistemic_term)

        return float(G)

    def select_action(self, qs: jnp.ndarray) -> int:
        """Select action that minimizes expected free energy.

        Args:
            qs: Current state beliefs

        Returns:
            Selected action index
        """
        expected_free_energies = []

        for action in range(self.model.n_actions):
            G = self.compute_expected_free_energy(action, qs)
            expected_free_energies.append(G)

        # Select action with minimum expected free energy
        best_action = jnp.argmin(jnp.array(expected_free_energies))

        return int(best_action)
```

### Complete Active Inference Agent
```python
class ActiveInferenceAgent:
    """Complete active inference agent implementation."""

    def __init__(self, n_states: int, n_actions: int, n_observations: int):
        """Initialize active inference agent.

        Args:
            n_states: Number of hidden states
            n_actions: Number of actions
            n_observations: Number of observations
        """
        self.model = GenerativeModel(n_states, n_actions, n_observations)
        self.inference = VariationalInference(self.model)
        self.action_selector = ActionSelection(self.model, self.inference)

        # Learning parameters
        self.learning_rate = 0.01
        self.observation_history = []

    def perceive(self, observation: int) -> Dict[str, jnp.ndarray]:
        """Perceive observation and update beliefs.

        Args:
            observation: Current observation

        Returns:
            Dictionary with perception results
        """
        self.observation_history.append(observation)

        # Perform variational inference
        qs = self.inference.infer_states(observation)

        # Compute variational free energy
        F = self.inference.compute_variational_free_energy(observation)

        return {
            'beliefs': qs,
            'free_energy': F,
            'observation': observation
        }

    def act(self, beliefs: jnp.ndarray) -> Dict[str, int]:
        """Select and execute action based on current beliefs.

        Args:
            beliefs: Current state beliefs

        Returns:
            Dictionary with action selection results
        """
        # Select action minimizing expected free energy
        action = self.action_selector.select_action(beliefs)

        # Compute expected free energies for all actions
        expected_free_energies = []
        for a in range(self.model.n_actions):
            G = self.action_selector.compute_expected_free_energy(a, beliefs)
            expected_free_energies.append(float(G))

        return {
            'action': action,
            'expected_free_energies': expected_free_energies
        }

    def learn(self, observation: int, action: int, next_observation: int):
        """Learn from experience using Bayesian model updating.

        Args:
            observation: Previous observation
            action: Action taken
            next_observation: Resulting observation
        """
        # Update likelihood matrix A using Bayesian updating
        # This is a simplified learning rule
        alpha = 1.0  # Learning rate

        # Find most likely previous state
        prev_state = jnp.argmax(self.inference.qs)

        # Update A: p(o|s) increases for observed transitions
        self.model.A = self.model.A.at[next_observation, prev_state].add(alpha)
        self.model.A = self.model.A / jnp.sum(self.model.A, axis=0)  # Normalize

        # Update transition matrix B: p(s'|s,a) increases for observed transitions
        next_state = jnp.argmax(self.inference.qs)  # Simplified
        self.model.B = self.model.B.at[next_state, prev_state, action].add(alpha)
        self.model.B = self.model.B / jnp.sum(self.model.B, axis=0)  # Normalize

    def step(self, observation: int) -> Dict:
        """Complete active inference step: perceive, act, learn.

        Args:
            observation: Current observation

        Returns:
            Complete step results
        """
        # Perceive
        perception = self.perceive(observation)

        # Act
        action_selection = self.act(perception['beliefs'])

        return {
            'perception': perception,
            'action': action_selection,
            'timestamp': len(self.observation_history)
        }
```

## 🎯 Practical Examples

### Grid World Navigation
```python
class GridWorldAgent(ActiveInferenceAgent):
    """Active inference agent for grid world navigation."""

    def __init__(self, grid_size: int = 5):
        """Initialize grid world agent.

        Args:
            grid_size: Size of square grid world
        """
        n_states = grid_size * grid_size  # Position states
        n_actions = 4  # Up, Down, Left, Right
        n_observations = grid_size * grid_size  # Position observations

        super().__init__(n_states, n_actions, n_observations)

        self.grid_size = grid_size
        self.position_to_state = lambda x, y: x * grid_size + y

    def get_observation_from_position(self, x: int, y: int) -> int:
        """Convert position to observation index."""
        return self.position_to_state(x, y)

    def get_position_from_state(self, state: int) -> Tuple[int, int]:
        """Convert state index to position."""
        x = state // self.grid_size
        y = state % self.grid_size
        return x, y

    def simulate_action(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Simulate action in grid world.

        Args:
            position: Current (x, y) position
            action: Action index (0: up, 1: down, 2: left, 3: right)

        Returns:
            New (x, y) position after action
        """
        x, y = position

        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.grid_size - 1, y + 1)

        return x, y

# Usage example
def run_grid_world_simulation():
    """Demonstrate grid world navigation."""
    agent = GridWorldAgent(grid_size=3)

    # Start at position (0, 0)
    current_position = (0, 0)
    goal_position = (2, 2)

    for step in range(10):
        # Get observation
        observation = agent.get_observation_from_position(*current_position)

        # Active inference step
        results = agent.step(observation)

        # Execute action
        action = results['action']['action']
        new_position = agent.simulate_action(current_position, action)

        print(f"Step {step}: Position {current_position} -> Action {action} -> Position {new_position}")
        print(f"  Beliefs: {results['perception']['beliefs'][:5]}...")  # Show first 5 beliefs
        print(f"  Free Energy: {results['perception']['free_energy']:.3f}")

        current_position = new_position

        # Check if goal reached
        if current_position == goal_position:
            print("Goal reached!")
            break
```

### Physiological Homeostasis
```python
class HomeostaticAgent(ActiveInferenceAgent):
    """Active inference agent modeling physiological homeostasis."""

    def __init__(self):
        """Initialize physiological agent."""
        n_states = 10  # Physiological states (e.g., glucose levels)
        n_actions = 3  # Actions: eat, rest, exercise
        n_observations = 10  # Physiological observations

        super().__init__(n_states, n_actions, n_observations)

        # Physiological parameters
        self.optimal_state = 5  # Target physiological state
        self.current_state = 5  # Current physiological state

    def update_physiology(self, action: int):
        """Update physiological state based on action.

        Args:
            action: Action taken (0: eat, 1: rest, 2: exercise)
        """
        if action == 0:  # Eat - increases physiological state
            self.current_state = min(9, self.current_state + 1)
        elif action == 1:  # Rest - maintains physiological state
            pass  # No change
        elif action == 2:  # Exercise - decreases physiological state
            self.current_state = max(0, self.current_state - 1)

        # Add random fluctuations
        self.current_state += np.random.normal(0, 0.5)
        self.current_state = np.clip(self.current_state, 0, 9)

    def get_observation(self) -> int:
        """Get current physiological observation."""
        return int(round(self.current_state))

# Usage example
def run_homeostasis_simulation():
    """Demonstrate physiological homeostasis."""
    agent = HomeostaticAgent()

    for step in range(20):
        # Get physiological observation
        observation = agent.get_observation()

        # Active inference step
        results = agent.step(observation)

        # Execute action
        action = results['action']['action']
        agent.update_physiology(action)

        action_names = ['Eat', 'Rest', 'Exercise']
        print(f"Step {step}: State {agent.current_state:.1f} -> {action_names[action]}")
        print(f"  Free Energy: {results['perception']['free_energy']:.3f}")

        # Check homeostasis
        deviation = abs(agent.current_state - agent.optimal_state)
        print(f"  Homeostatic deviation: {deviation:.3f}")
```

## 🔧 Implementation Tools and Libraries

### Core Dependencies
```python
# requirements.txt
jax>=0.4.0
jaxlib>=0.4.0
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
```

### Utility Functions
```python
def plot_beliefs(beliefs: jnp.ndarray, title: str = "Beliefs"):
    """Plot belief distribution."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(beliefs)), beliefs)
    plt.xlabel("State")
    plt.ylabel("Belief Probability")
    plt.title(title)
    plt.show()

def plot_free_energy_history(history: List[float]):
    """Plot variational free energy over time."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.xlabel("Time Step")
    plt.ylabel("Variational Free Energy")
    plt.title("Free Energy Minimization")
    plt.show()

def save_agent_state(agent: ActiveInferenceAgent, filename: str):
    """Save agent state to file."""
    state = {
        'A': agent.model.A,
        'B': agent.model.B,
        'C': agent.model.C,
        'D': agent.model.D,
        'qs': agent.inference.qs
    }
    np.savez(filename, **state)

def load_agent_state(agent: ActiveInferenceAgent, filename: str):
    """Load agent state from file."""
    state = np.load(filename)
    agent.model.A = jnp.array(state['A'])
    agent.model.B = jnp.array(state['B'])
    agent.model.C = jnp.array(state['C'])
    agent.model.D = jnp.array(state['D'])
    agent.inference.qs = jnp.array(state['qs'])
```

## 🧪 Testing and Validation

### Unit Tests
```python
import pytest

def test_generative_model_initialization():
    """Test generative model initialization."""
    model = GenerativeModel(n_states=3, n_actions=2, n_observations=3)

    assert model.A.shape == (3, 3)
    assert model.B.shape == (3, 3, 2)
    assert model.C.shape == (3,)
    assert model.D.shape == (3,)

def test_variational_inference():
    """Test variational inference."""
    model = GenerativeModel(n_states=3, n_actions=2, n_observations=3)
    inference = VariationalInference(model)

    observation = 1
    beliefs = inference.infer_states(observation)

    assert len(beliefs) == 3
    assert jnp.sum(beliefs) == pytest.approx(1.0, abs=1e-6)

def test_action_selection():
    """Test action selection."""
    model = GenerativeModel(n_states=3, n_actions=2, n_observations=3)
    inference = VariationalInference(model)
    action_selector = ActionSelection(model, inference)

    beliefs = jnp.array([0.5, 0.3, 0.2])
    action = action_selector.select_action(beliefs)

    assert 0 <= action < 2
```

### Performance Benchmarks
```python
def benchmark_inference_speed():
    """Benchmark inference speed."""
    import time

    model = GenerativeModel(n_states=100, n_actions=10, n_observations=50)
    inference = VariationalInference(model)

    observation = 25
    n_iterations = 100

    start_time = time.time()
    for _ in range(n_iterations):
        inference.infer_states(observation)
    end_time = time.time()

    avg_time = (end_time - start_time) / n_iterations
    print(f"Average inference time: {avg_time:.4f} seconds")

def benchmark_free_energy_minimization():
    """Test free energy minimization over learning."""
    agent = ActiveInferenceAgent(n_states=5, n_actions=3, n_observations=5)

    free_energy_history = []

    for episode in range(100):
        observation = np.random.randint(5)
        results = agent.step(observation)
        free_energy_history.append(results['perception']['free_energy'])

        # Simulate learning (simplified)
        next_observation = np.random.randint(5)
        agent.learn(observation, results['action']['action'], next_observation)

    # Check if free energy decreases over time
    initial_avg = np.mean(free_energy_history[:20])
    final_avg = np.mean(free_energy_history[-20:])

    assert final_avg < initial_avg, "Free energy should decrease with learning"
```

## 🚀 Advanced Implementations

### Hierarchical Active Inference
```python
class HierarchicalActiveInferenceAgent:
    """Hierarchical active inference agent with multiple timescales."""

    def __init__(self, hierarchy_levels: List[int], n_actions: int, n_observations: int):
        """Initialize hierarchical agent.

        Args:
            hierarchy_levels: List of state counts for each level
            n_actions: Number of actions
            n_observations: Number of observations
        """
        self.levels = []
        for i, n_states in enumerate(hierarchy_levels):
            level_agent = ActiveInferenceAgent(n_states, n_actions, n_observations)
            self.levels.append(level_agent)

        # Inter-level connections
        self.level_couplings = self.initialize_level_couplings()

    def initialize_level_couplings(self):
        """Initialize couplings between hierarchical levels."""
        couplings = []
        for i in range(len(self.levels) - 1):
            # Coupling matrices between adjacent levels
            coupling = jnp.eye(self.levels[i].model.n_states, self.levels[i+1].model.n_states)
            couplings.append(coupling)
        return couplings

    def hierarchical_inference(self, observation: int):
        """Perform hierarchical inference across levels."""
        # Bottom-up inference
        for level in self.levels:
            level.perceive(observation)

        # Top-down prediction and error propagation
        for i in reversed(range(len(self.levels) - 1)):
            # Propagate predictions downward
            higher_beliefs = self.levels[i+1].inference.qs
            lower_predictions = self.level_couplings[i] @ higher_beliefs

            # Update lower level with top-down predictions
            self.levels[i].inference.qs = 0.5 * self.levels[i].inference.qs + 0.5 * lower_predictions
            self.levels[i].inference.qs = self.levels[i].inference.qs / jnp.sum(self.levels[i].inference.qs)
```

### Multi-Agent Active Inference
```python
class MultiAgentActiveInference:
    """Multi-agent active inference system."""

    def __init__(self, n_agents: int, n_states: int, n_actions: int, n_observations: int):
        """Initialize multi-agent system."""
        self.agents = [
            ActiveInferenceAgent(n_states, n_actions, n_observations)
            for _ in range(n_agents)
        ]

        # Communication channels
        self.communication_matrix = jnp.ones((n_agents, n_agents)) * 0.1
        self.communication_matrix = self.communication_matrix.at[jnp.diag_indices(n_agents)].set(0.0)

    def multi_agent_step(self, observations: List[int]) -> List[Dict]:
        """Perform multi-agent active inference step."""
        # Individual agent steps
        individual_results = []
        for i, (agent, obs) in enumerate(zip(self.agents, observations)):
            result = agent.step(obs)
            individual_results.append(result)

        # Inter-agent communication
        self.communicate_beliefs()

        return individual_results

    def communicate_beliefs(self):
        """Exchange beliefs between agents."""
        for i, agent_i in enumerate(self.agents):
            received_beliefs = []

            for j, agent_j in enumerate(self.agents):
                if i != j:
                    # Weighted communication
                    weight = self.communication_matrix[i, j]
                    received_beliefs.append(weight * agent_j.inference.qs)

            if received_beliefs:
                # Average received beliefs
                avg_received = jnp.mean(jnp.stack(received_beliefs), axis=0)

                # Update agent beliefs with social information
                social_influence = 0.3  # Social influence strength
                agent_i.inference.qs = (1 - social_influence) * agent_i.inference.qs + social_influence * avg_received
                agent_i.inference.qs = agent_i.inference.qs / jnp.sum(agent_i.inference.qs)
```

## 📚 References and Extensions

### Core Implementation References
- [[mathematics/free_energy_principle|Free Energy Principle]]
- [[cognitive/active_inference|Active Inference Theory]]
- [[tools/src/models/|Existing Implementation Tools]]

### Advanced Topics
- **Hierarchical Active Inference**: Multi-timescale inference
- **Multi-Agent Systems**: Collective active inference
- **Continuous State Spaces**: Differential equation formulations
- **Quantum Active Inference**: Quantum probability extensions

### Performance Optimization
- **GPU Acceleration**: CUDA implementations for large-scale models
- **Distributed Computing**: Parallel inference across multiple machines
- **Approximate Inference**: Variational approximations for complex models
- **Online Learning**: Continuous model updating during deployment

---

> **Practical Implementation**: Provides working code examples for active inference applications.

---

> **Modular Design**: Components can be mixed and matched for different use cases.

---

> **Extensible Framework**: Easy to extend for hierarchical and multi-agent scenarios.

---

> **Research Enablement**: Accelerates implementation of active inference in real systems.
