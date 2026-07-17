---
title: "Simulation Environments for Active Inference"
type: implementation
status: stable
created: 2025-02-06
updated: 2025-02-06
tags:
  - free_energy_principle
  - implementation
  - simulation
  - environments
  - openai_gym
  - grid_world
  - benchmarking
  - visualization
semantic_relations:
  - type: relates
    links:
      - [[python_framework|Python Framework]]
      - [[neural_networks|Neural Network Implementations]]
      - [[robotics|Robotics Implementations]]
      - [[benchmarking|Benchmarking]]
      - [[knowledge_base/free_energy_principle/mathematics/expected_free_energy|Expected Free Energy]]
      - [[knowledge_base/free_energy_principle/cognitive/decision_making|Decision Making]]
---

# Simulation Environments for Active Inference

## Overview

Simulation environments provide the essential testbed for developing, validating, and comparing active inference agents. Unlike reinforcement learning, where environments supply reward signals, active inference agents require environments that provide observations from which the agent infers hidden states and selects actions to minimize expected free energy. This distinction shapes how environments are designed and how agent-environment interfaces are structured.

This document covers the design and implementation of simulation environments for active inference, from simple grid worlds to continuous control domains, with emphasis on proper integration, logging, visualization, and reproducibility.

## Architecture and Design

### Environment Requirements for Active Inference

Active inference agents differ from standard RL agents in their interface requirements:

| Aspect | RL Environment | Active Inference Environment |
|--------|---------------|------------------------------|
| Agent receives | Observation + Reward | Observation (possibly partial) |
| Agent's objective | Maximize cumulative reward | Minimize (expected) free energy |
| Hidden states | Optional (POMDP) | Essential (the agent infers these) |
| Observation model | Implicit | Explicit $p(o \mid s)$ needed for model design |
| Transition model | Implicit | Explicit $p(s' \mid s, a)$ needed for model design |
| Reward | Provided by env | Encoded as prior preferences $p(o)$ |

The key implication: when designing environments for active inference, we must clearly specify the generative process (how the environment works) separately from the generative model (what the agent believes).

### Generative Process vs. Generative Model

$$
\text{Generative Process:} \quad s_{t+1} \sim p^*(s_{t+1} | s_t, a_t), \quad o_t \sim p^*(o_t | s_t)
$$

$$
\text{Generative Model:} \quad \tilde{s}_{t+1} \sim p(s_{t+1} | s_t, a_t; \theta), \quad \tilde{o}_t \sim p(o_t | s_t; \theta)
$$

When these diverge, the agent experiences surprise, driving learning ($\theta$ updates) and action (minimize $\mathcal{F}$). The mismatch between process and model is the engine of adaptive behavior.

### Environment Taxonomy

**Discrete State Spaces:**
- Grid worlds (navigation, foraging, T-maze)
- Discrete MDPs/POMDPs
- Graph-structured environments

**Continuous State Spaces:**
- Cart-pole, pendulum, mountain car
- Robotic manipulation
- Vehicle navigation

**Multi-Agent:**
- Predator-prey
- Cooperative navigation
- Communication games

**Hierarchical:**
- Rooms with sub-goals
- Multi-step manipulation
- Temporal abstraction tasks

## Implementation Details

### Discrete Grid World for Active Inference

The grid world is the canonical testbed for discrete active inference. Key design choices:

- **State space**: Grid positions (and optionally cue/context states)
- **Observation space**: Position observations (partial observability via limited vision)
- **Action space**: Up, Down, Left, Right (and optionally Stay)
- **Prior preferences**: Preferred positions encoded as $\log p(o)$ over observation outcomes

```python
import numpy as np
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces


class ActiveInferenceGridWorld(gym.Env):
    """
    Grid world environment designed for active inference agents.

    Provides:
    - Explicit observation likelihood matrix A (for model specification)
    - Explicit transition matrix B (for model specification)
    - Partial observability via limited sensory range
    - Configurable prior preference landscapes

    Unlike standard RL grid worlds, this environment exposes the generative
    process matrices so agents can use them as the generative model
    (or learn mismatched models for more interesting experiments).
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"]}

    def __init__(self, grid_size: int = 5, n_cues: int = 2,
                 sensory_range: int = 1, render_mode: str = "ansi"):
        super().__init__()
        self.grid_size = grid_size
        self.n_cues = n_cues
        self.sensory_range = sensory_range
        self.render_mode = render_mode

        self.n_positions = grid_size * grid_size
        self.n_states = self.n_positions  # Can extend with context states
        self.n_actions = 5  # Up, Down, Left, Right, Stay

        # Gym spaces
        self.observation_space = spaces.Dict({
            "position": spaces.Discrete(self.n_positions),
            "local_view": spaces.Box(
                low=0, high=1,
                shape=(2 * sensory_range + 1, 2 * sensory_range + 1),
                dtype=np.float32
            ),
        })
        self.action_space = spaces.Discrete(self.n_actions)

        # Build generative process matrices
        self._build_likelihood_matrix()
        self._build_transition_matrices()

        # Place cues and goal
        self.cue_positions = []
        self.goal_position = None
        self.agent_position = None
        self._setup_task()

    def _build_likelihood_matrix(self):
        """
        Build observation likelihood matrix A.
        A[o, s] = p(o | s) -- probability of observation o given state s.
        For a fully observable grid, A = I.
        For partial observability, A encodes sensory ambiguity.
        """
        self.A = np.eye(self.n_states)  # Start with full observability
        # Add noise for partial observability
        noise_level = 0.05
        self.A = (1 - noise_level) * self.A + noise_level / self.n_states

    def _build_transition_matrices(self):
        """
        Build transition matrices B[a].
        B[s', s, a] = p(s' | s, a) -- probability of transitioning to s'
        from s under action a.
        """
        self.B = np.zeros((self.n_actions, self.n_states, self.n_states))

        for s in range(self.n_states):
            row, col = divmod(s, self.grid_size)
            for a in range(self.n_actions):
                new_row, new_col = row, col
                if a == 0:    # Up
                    new_row = max(0, row - 1)
                elif a == 1:  # Down
                    new_row = min(self.grid_size - 1, row + 1)
                elif a == 2:  # Left
                    new_col = max(0, col - 1)
                elif a == 3:  # Right
                    new_col = min(self.grid_size - 1, col + 1)
                # a == 4: Stay

                new_s = new_row * self.grid_size + new_col

                # Deterministic transitions with small slip probability
                self.B[a, new_s, s] = 0.9
                self.B[a, s, s] += 0.1  # Slip: stay in place

        # Normalize columns
        for a in range(self.n_actions):
            col_sums = self.B[a].sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            self.B[a] /= col_sums

    def _setup_task(self):
        """Initialize cue and goal positions."""
        positions = np.random.choice(
            self.n_positions, size=self.n_cues + 1, replace=False
        )
        self.cue_positions = positions[:self.n_cues].tolist()
        self.goal_position = positions[self.n_cues]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_position = np.random.randint(self.n_positions)
        self._setup_task()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple:
        # Sample next state from transition matrix
        transition_probs = self.B[action, :, self.agent_position]
        self.agent_position = np.random.choice(
            self.n_states, p=transition_probs
        )

        obs = self._get_obs()
        info = self._get_info()

        # For compatibility: reward = 1 at goal, 0 otherwise
        reward = 1.0 if self.agent_position == self.goal_position else 0.0
        terminated = self.agent_position == self.goal_position
        truncated = False

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> Dict:
        """Generate observation from current state."""
        # Sample from likelihood
        obs_probs = self.A[:, self.agent_position]
        observed_position = np.random.choice(self.n_states, p=obs_probs)

        # Generate local view
        row, col = divmod(self.agent_position, self.grid_size)
        local_view = np.zeros(
            (2 * self.sensory_range + 1, 2 * self.sensory_range + 1),
            dtype=np.float32
        )
        for dr in range(-self.sensory_range, self.sensory_range + 1):
            for dc in range(-self.sensory_range, self.sensory_range + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    pos = r * self.grid_size + c
                    if pos in self.cue_positions:
                        local_view[dr + self.sensory_range,
                                   dc + self.sensory_range] = 0.5
                    elif pos == self.goal_position:
                        local_view[dr + self.sensory_range,
                                   dc + self.sensory_range] = 1.0

        return {"position": observed_position, "local_view": local_view}

    def _get_info(self) -> Dict:
        """Return ground-truth info for logging/analysis."""
        return {
            "true_position": self.agent_position,
            "goal_position": self.goal_position,
            "cue_positions": self.cue_positions,
            "distance_to_goal": self._manhattan_distance(
                self.agent_position, self.goal_position
            ),
        }

    def _manhattan_distance(self, pos1: int, pos2: int) -> int:
        r1, c1 = divmod(pos1, self.grid_size)
        r2, c2 = divmod(pos2, self.grid_size)
        return abs(r1 - r2) + abs(c1 - c2)

    def get_generative_process(self) -> Dict:
        """
        Expose the generative process matrices.
        Agents can use these directly (matched model) or learn their own
        (mismatched model) for more realistic experiments.
        """
        return {"A": self.A.copy(), "B": self.B.copy()}

    def get_preference_prior(self, sharpness: float = 2.0) -> np.ndarray:
        """
        Generate a prior preference distribution C over observations.
        C[o] = log p(o) -- log probability of preferred observations.
        High values at goal, low elsewhere.
        """
        C = np.full(self.n_states, -sharpness)
        C[self.goal_position] = 0.0  # Goal is the most preferred
        for cue_pos in self.cue_positions:
            C[cue_pos] = -sharpness / 2  # Cues are somewhat preferred
        return C

    def render(self):
        if self.render_mode == "ansi":
            grid = [["." for _ in range(self.grid_size)]
                    for _ in range(self.grid_size)]
            for cp in self.cue_positions:
                r, c = divmod(cp, self.grid_size)
                grid[r][c] = "C"
            gr, gc = divmod(self.goal_position, self.grid_size)
            grid[gr][gc] = "G"
            ar, ac = divmod(self.agent_position, self.grid_size)
            grid[ar][ac] = "A"
            return "\n".join(" ".join(row) for row in grid)
```

### Simulation Harness with Logging

```python
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable


@dataclass
class StepLog:
    """Record of a single simulation step."""
    step: int
    observation: Any
    action: int
    true_state: Any
    free_energy: float = 0.0
    efe: float = 0.0
    belief_entropy: float = 0.0
    distance_to_goal: int = 0
    elapsed_ms: float = 0.0


@dataclass
class EpisodeLog:
    """Record of a full episode."""
    episode: int
    steps: List[StepLog] = field(default_factory=list)
    total_reward: float = 0.0
    reached_goal: bool = False
    total_free_energy: float = 0.0
    wall_time_s: float = 0.0


class SimulationHarness:
    """
    Simulation harness for running and logging active inference experiments.

    Features:
    - Structured logging of every step (observations, beliefs, actions, metrics)
    - Episode-level and run-level statistics
    - Reproducibility via seed management
    - Checkpoint saving and loading
    - Callback hooks for custom visualization and analysis
    """

    def __init__(self, env, agent, log_dir: str = "./logs",
                 seed: int = 42):
        self.env = env
        self.agent = agent
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.callbacks: List[Callable] = []
        self.episode_logs: List[EpisodeLog] = []

    def add_callback(self, callback: Callable):
        """Add a callback function called after each step."""
        self.callbacks.append(callback)

    def run_episode(self, episode_num: int, max_steps: int = 200,
                    render: bool = False) -> EpisodeLog:
        """Run a single episode and return structured log."""
        episode_log = EpisodeLog(episode=episode_num)
        episode_start = time.time()

        obs, info = self.env.reset(seed=self.seed + episode_num)
        self.agent.reset()

        for step in range(max_steps):
            step_start = time.time()

            # Agent perceives and acts
            action, agent_info = self.agent.step(obs)

            # Environment transitions
            next_obs, reward, terminated, truncated, env_info = \
                self.env.step(action)

            # Log step
            step_log = StepLog(
                step=step,
                observation=int(obs["position"]) if isinstance(obs, dict) else obs,
                action=int(action),
                true_state=env_info.get("true_position", None),
                free_energy=agent_info.get("free_energy", 0.0),
                efe=agent_info.get("efe", 0.0),
                belief_entropy=agent_info.get("belief_entropy", 0.0),
                distance_to_goal=env_info.get("distance_to_goal", 0),
                elapsed_ms=(time.time() - step_start) * 1000,
            )
            episode_log.steps.append(step_log)
            episode_log.total_reward += reward
            episode_log.total_free_energy += step_log.free_energy

            # Callbacks
            for callback in self.callbacks:
                callback(step_log, episode_log, env_info, agent_info)

            if render:
                print(self.env.render())
                print(f"Step {step}: action={action}, "
                      f"F={step_log.free_energy:.3f}, "
                      f"dist={step_log.distance_to_goal}")

            obs = next_obs
            if terminated or truncated:
                episode_log.reached_goal = terminated
                break

        episode_log.wall_time_s = time.time() - episode_start
        self.episode_logs.append(episode_log)
        return episode_log

    def run_experiment(self, n_episodes: int = 100, max_steps: int = 200,
                       render: bool = False) -> Dict:
        """Run a full experiment and return summary statistics."""
        for ep in range(n_episodes):
            episode_log = self.run_episode(ep, max_steps, render=render)
            if ep % 10 == 0:
                success_rate = np.mean([
                    e.reached_goal for e in self.episode_logs[-10:]
                ]) if len(self.episode_logs) >= 10 else 0.0
                avg_steps = np.mean([
                    len(e.steps) for e in self.episode_logs[-10:]
                ]) if len(self.episode_logs) >= 10 else 0.0
                print(f"Episode {ep}: success_rate={success_rate:.2f}, "
                      f"avg_steps={avg_steps:.1f}")

        return self.compute_summary()

    def compute_summary(self) -> Dict:
        """Compute summary statistics across all episodes."""
        return {
            "n_episodes": len(self.episode_logs),
            "success_rate": np.mean([e.reached_goal for e in self.episode_logs]),
            "avg_steps": np.mean([len(e.steps) for e in self.episode_logs]),
            "std_steps": np.std([len(e.steps) for e in self.episode_logs]),
            "avg_free_energy": np.mean([
                e.total_free_energy for e in self.episode_logs
            ]),
            "avg_wall_time_s": np.mean([
                e.wall_time_s for e in self.episode_logs
            ]),
        }

    def save_logs(self, filename: str = "experiment_log.json"):
        """Save all episode logs to JSON."""
        log_path = self.log_dir / filename
        data = {
            "seed": self.seed,
            "summary": self.compute_summary(),
            "episodes": [
                {
                    "episode": e.episode,
                    "total_reward": e.total_reward,
                    "reached_goal": e.reached_goal,
                    "total_free_energy": e.total_free_energy,
                    "wall_time_s": e.wall_time_s,
                    "n_steps": len(e.steps),
                }
                for e in self.episode_logs
            ],
        }
        with open(log_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Logs saved to {log_path}")
```

## Comparison of Simulation Platforms

| Platform | State Space | Observation | Physics | Active Inference Suitability |
|----------|------------|-------------|---------|------------------------------|
| Custom Grid World | Discrete | Discrete/Tabular | None | Excellent -- full control over generative process |
| Gymnasium (Gym) | Discrete/Continuous | Varied | Basic | Good -- standard interface, many environments |
| PyBullet | Continuous | Images/Vectors | Rigid body | Good -- free, good for robotics |
| MuJoCo | Continuous | Images/Vectors | Advanced contact | Excellent -- standard for continuous control |
| dm_control | Continuous | Images/Vectors | MuJoCo-based | Excellent -- well-designed benchmarks |
| PettingZoo | Multi-agent | Varied | Varied | Good -- for multi-agent active inference |
| Minigrid | Discrete | Partial obs images | None | Good -- partial observability built in |
| Custom Continuous | Continuous | Configurable | Custom | Excellent -- tailor to FEP needs |

### Platform Selection Guidance

- **Theoretical research** (discrete, small scale): Custom grid worlds or pymdp environments. Full control over matrices A, B, C, D.
- **Scaling experiments** (discrete, larger): Minigrid or custom POMDPs. Partial observability and configurable complexity.
- **Continuous control** (standard benchmarks): MuJoCo / dm_control. Well-established baselines for comparison.
- **Robotics prototyping**: PyBullet (free) or MuJoCo. Transfer to real robots via sim-to-real methods.
- **Multi-agent studies**: PettingZoo. Standard multi-agent interface with many built-in environments.
- **Custom research**: Build your own using the Gymnasium interface. Maximum flexibility.

## Visualization Tools

### Real-Time Belief Visualization

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class BeliefVisualizer:
    """
    Real-time visualization of agent beliefs during simulation.

    Displays:
    - Grid world state (agent, goal, cues)
    - Agent's belief distribution over states
    - Free energy over time
    - Expected free energy for each action
    """

    def __init__(self, env, figsize=(14, 5)):
        self.env = env
        self.fig, self.axes = plt.subplots(1, 3, figsize=figsize)
        self.fe_history = []
        self.belief_history = []

    def update(self, step_log: StepLog, beliefs: np.ndarray,
               efe_per_action: np.ndarray):
        """Update visualization with current step data."""
        grid_size = self.env.grid_size

        # Panel 1: Grid world state
        ax = self.axes[0]
        ax.clear()
        grid = np.zeros((grid_size, grid_size))
        # Show belief as heatmap
        for s in range(self.env.n_states):
            r, c = divmod(s, grid_size)
            grid[r, c] = beliefs[s]
        ax.imshow(grid, cmap='Blues', vmin=0, vmax=1)
        # Mark agent, goal
        ar, ac = divmod(step_log.true_state, grid_size)
        ax.plot(ac, ar, 'ro', markersize=12, label='Agent')
        gr, gc = divmod(self.env.goal_position, grid_size)
        ax.plot(gc, gr, 'g*', markersize=15, label='Goal')
        ax.set_title(f"Belief Map (step {step_log.step})")
        ax.legend(loc='upper right', fontsize=8)

        # Panel 2: Free energy over time
        self.fe_history.append(step_log.free_energy)
        ax = self.axes[1]
        ax.clear()
        ax.plot(self.fe_history, 'b-', linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Free Energy")
        ax.set_title("Variational Free Energy")

        # Panel 3: EFE per action
        ax = self.axes[2]
        ax.clear()
        action_labels = ["Up", "Down", "Left", "Right", "Stay"]
        colors = ['#e74c3c' if a == step_log.action else '#3498db'
                  for a in range(len(efe_per_action))]
        ax.bar(action_labels, -efe_per_action, color=colors)
        ax.set_ylabel("Negative EFE (higher = preferred)")
        ax.set_title("Action Selection")

        self.fig.tight_layout()
        plt.pause(0.01)
```

## Reproducibility

### Seed Management

Reproducibility in active inference experiments requires controlling randomness at multiple levels:

```python
def set_global_seed(seed: int):
    """Set seeds for all sources of randomness."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For full reproducibility (may reduce performance):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### Experiment Configuration

```python
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """Complete experiment configuration for reproducibility."""
    # Environment
    env_name: str = "ActiveInferenceGridWorld"
    grid_size: int = 5
    sensory_range: int = 1
    n_cues: int = 2

    # Agent
    agent_type: str = "discrete_active_inference"
    planning_horizon: int = 4
    policy_length: int = 3
    gamma: float = 16.0  # Precision on action selection

    # Experiment
    n_episodes: int = 100
    max_steps_per_episode: int = 200
    seed: int = 42

    # Logging
    log_dir: str = "./logs"
    save_beliefs: bool = False
    save_every_n_episodes: int = 10
```

## Best Practices

### Environment Design

1. **Expose generative process matrices** (A, B for discrete; dynamics functions for continuous) so agents can start with matched models or learn mismatched ones.
2. **Separate reward from observations**: Active inference agents do not use reward signals. Provide prior preferences $C$ instead, or let the agent define them.
3. **Partial observability is essential**: Full observability removes the need for state inference, which is a core part of active inference. Always include some form of partial observability.
4. **Provide ground-truth info for logging**: Return true states and other metadata in the `info` dict for analysis, even though the agent should not access them.

### Simulation Fidelity

1. **Start simple, scale up**: Validate agent behavior on 3x3 or 5x5 grids before moving to larger or continuous environments.
2. **Control stochasticity**: Environments should have configurable transition noise and observation noise to study robustness.
3. **Match environment to theory**: Ensure the environment's generative process can actually be approximated by the agent's generative model class.

### Logging and Analysis

1. **Log beliefs, not just actions**: The internal belief dynamics are often more informative than external behavior for understanding active inference.
2. **Track free energy components separately**: Log complexity and accuracy independently to diagnose whether the agent is underfitting (high accuracy loss) or overfitting (high complexity).
3. **Record wall-clock time**: Computational cost per step is a key metric for practical deployment.
4. **Save full episode trajectories** for post-hoc analysis and visualization.

## References

1. Heins, C., Millidge, B., Demekas, D., Klein, B., Friston, K., Couzin, I. D., & Tschantz, A. (2022). pymdp: A Python library for active inference in discrete state spaces. *Journal of Open Source Software*, 7(73), 4098.
2. Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). OpenAI Gym. *arXiv preprint* arXiv:1606.01540.
3. noterov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A physics engine for model-based control. *IEEE/RSJ International Conference on Intelligent Robots and Systems*.
4. Coumans, E., & Bai, Y. (2016-2021). PyBullet, a Python module for physics simulation for games, robotics and machine learning. http://pybullet.org
5. Tassa, Y., et al. (2018). DeepMind Control Suite. *arXiv preprint* arXiv:1801.00690.
6. Terry, J. K., et al. (2021). PettingZoo: Gym for multi-agent reinforcement learning. *NeurIPS*.
7. Chevalier-Boisvert, M., Willems, L., & Pal, S. (2018). Minimalistic gridworld environment for Gymnasium. https://github.com/Farama-Foundation/Minigrid
8. Fountas, Z., Sajid, N., Mediano, P. A. M., & Friston, K. (2020). Deep active inference agents using Monte-Carlo methods. *NeurIPS*.

## See Also

- [[python_framework|Python Framework]] -- base implementations for agents running in these environments
- [[neural_networks|Neural Network Implementations]] -- deep active inference agents that use these environments
- [[robotics|Robotics Implementations]] -- transferring simulation results to real robots
- [[benchmarking|Benchmarking]] -- systematic evaluation using these environments
- [[knowledge_base/free_energy_principle/mathematics/expected_free_energy|Expected Free Energy]] -- the objective function driving action selection
- [[knowledge_base/free_energy_principle/cognitive/decision_making|Decision Making]] -- the cognitive theory behind environment interaction
