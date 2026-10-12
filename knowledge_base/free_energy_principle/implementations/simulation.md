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
      - "[[python_framework|Python Framework]]"
      - "[[neural_networks|Neural Network Implementations]]"
      - "[[robotics|Robotics Implementations]]"
      - "[[benchmarking|Benchmarking]]"
      - "[[knowledge_base/free_energy_principle/mathematics/expected_free_energy|Expected Free Energy]]"
      - "[[knowledge_base/free_energy_principle/cognitive/decision_making|Decision Making]]"
---

# Simulation Environments for Active Inference

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

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

size = 3                                     # 3x3 grid, illustrative
n = size * size
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right

# Generative process: B[a][s', s] moves deterministically, walls leave the state unchanged
B = np.zeros((4, n, n))
for a, (dr, dc) in enumerate(moves):
    for s in range(n):
        r, c = divmod(s, size)
        r2, c2 = min(max(r + dr, 0), size - 1), min(max(c + dc, 0), size - 1)
        B[a, r2 * size + c2, s] = 1.0

# Noisy position observations: p(o | s) puts 0.9 on the true position
A = np.full((n, n), 0.1 / (n - 1))
np.fill_diagonal(A, 0.9)

# Prior preferences log p(o): the bottom-right cell is preferred
log_C = np.full(n, -2.0)
log_C[-1] = 0.0

rng = np.random.default_rng(0)
s = 0
for a in (3, 3, 1, 1):                       # right, right, down, down
    s = int(np.argmax(B[a][:, s]))
    o = rng.choice(n, p=A[:, s])
    print("action", a, "state", s, "observation", o)
```

### Simulation Harness with Logging

```python
import json
import numpy as np

rng = np.random.default_rng(0)

def run_episode(step, act, s0, max_steps=20):
    """Collect one dict per step, keeping true states out of the agent's reach."""
    log, s, o = [], s0, s0
    for t in range(max_steps):
        a = act(o)
        s, o, done = step(s, a)
        log.append({"t": t, "action": int(a), "true_state": int(s), "obs": int(o)})
        if done:
            break
    return log

# Toy five-state corridor, illustrative: action 1 moves right, 0 moves left
def step(s, a):
    s = int(np.clip(s + (1 if a == 1 else -1), 0, 4))
    return s, s, s == 4

log = run_episode(step, lambda o: rng.integers(2), s0=0, max_steps=200)
print(len(log), json.dumps(log[-1]))
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


## Reproducibility

### Seed Management

Reproducibility in active inference experiments requires controlling randomness at multiple levels:

```python
import numpy as np

seed = 1234                                  # one seed in the experiment config
env_seed, agent_seed = np.random.SeedSequence(seed).spawn(2)
env_rng = np.random.default_rng(env_seed)
agent_rng = np.random.default_rng(agent_seed)

# Independent streams, reproducible from the single seed
print(env_rng.integers(100, size=3), agent_rng.integers(100, size=3))
```

### Experiment Configuration

```python
import json
from dataclasses import asdict, dataclass

@dataclass
class ExperimentConfig:
    seed: int = 1234
    grid_size: int = 5
    n_episodes: int = 100
    max_steps: int = 200
    save_beliefs: bool = False

config = ExperimentConfig(grid_size=3)
print(json.dumps(asdict(config)))            # store alongside the logs
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
