---
title: Custom Control Modes
type: concept
status: stable
created: 2026-02-06
updated: 2026-02-07
tags:
  - control-modes
  - agent-configuration
  - adaptive-control
  - mode-switching
  - active-inference
semantic_relations:
  - type: relates
    links:
      - [[active_inference_for_control]]
      - [[advanced_control]]
      - [[basic_homeostatic_control]]
      - [[inference_configuration]]
---

# Custom Control Modes

## Overview

Custom control modes allow Active Inference agents to dynamically switch between different inference and control strategies depending on context. A control mode defines a configuration of precision weights, planning horizons, and learning rates optimized for specific task demands — analogous to how biological organisms switch between fight-or-flight, rest-and-digest, and exploration modes.

## Control Mode Architecture

### Mode Definition

> [!note] Illustrative pseudocode
> The class below is a conceptual sketch; the installed package's abstract
> `cognitive.models.active_inference.ControlMode` contract differs (it
> requires `compute_policy_prior(state, goal)`).

```python
class ModeConfig:
    def __init__(self, name, gamma, planning_depth, lr, precision_over_obs):
        self.name = name
        self.gamma = gamma                    # Policy precision
        self.planning_depth = planning_depth  # How far ahead to plan
        self.lr = lr                          # Learning rate
        self.precision_obs = precision_over_obs  # Sensory precision
    
    def apply(self, agent):
        agent.gamma = self.gamma
        agent.T = self.planning_depth
        agent.learning_rate = self.lr
        agent.sensory_precision = self.precision_obs

# Pre-defined modes
EXPLORE_MODE = ModeConfig("explore", gamma=1.0, planning_depth=1, lr=1.0, precision_over_obs=0.5)
EXPLOIT_MODE = ModeConfig("exploit", gamma=32.0, planning_depth=5, lr=0.01, precision_over_obs=2.0)
LEARN_MODE = ModeConfig("learn", gamma=4.0, planning_depth=2, lr=2.0, precision_over_obs=1.0)
EMERGENCY_MODE = ModeConfig("emergency", gamma=64.0, planning_depth=1, lr=0.0, precision_over_obs=4.0)
```

### Mode Switching Logic

```python
class ModeController:
    def __init__(self, modes, threshold_entropy=1.5, threshold_error=2.0):
        self.modes = modes
        self.entropy_threshold = threshold_entropy
        self.error_threshold = threshold_error
    
    def select_mode(self, agent_state):
        entropy = compute_belief_entropy(agent_state['beliefs'])
        pe_magnitude = np.mean(agent_state['prediction_errors'])
        
        if pe_magnitude > self.error_threshold:
            return self.modes['emergency']
        elif entropy > self.entropy_threshold:
            return self.modes['explore']
        elif agent_state['is_learning']:
            return self.modes['learn']
        else:
            return self.modes['exploit']
```

## Mode Comparison

| Mode | $\gamma$ | Depth | $\eta$ | Character |
| --- | --- | --- | --- | --- |
| Explore | Low | Short | High | Curious, information-seeking |
| Exploit | High | Long | Low | Goal-directed, efficient |
| Learn | Medium | Medium | Very high | Rapidly updating models |
| Emergency | Very high | Minimal | None | Fast, deterministic response |
| Cruise | Medium | Medium | Low | Steady-state, stable |

## Biological Analogues

| Mode | Biological Equivalent | Neuromodulator |
| --- | --- | --- |
| Explore | Curiosity, play | Dopamine (tonic) |
| Exploit | Goal pursuit | Dopamine (phasic) |
| Emergency | Fight-or-flight | Noradrenaline |
| Rest | Rest-and-digest | Acetylcholine |

## Related Topics

- [[active_inference_for_control]] — Active Inference control
- [[advanced_control]] — Advanced control methods
- [[basic_homeostatic_control]] — Homeostatic control
- [[inference_configuration]] — Inference parameters
- [[precision_weighting]] — Precision modulation
