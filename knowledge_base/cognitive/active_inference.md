---
title: Active Inference
type: concept
status: stable
created: 2024-02-12
tags:
  - cognitive
  - neuroscience
  - computation
semantic_relations:
  - type: foundation
    links: 
      - [[free_energy_principle]]
      - [[variational_inference]]
  - type: relates
    links:
      - [[predictive_coding]]
      - [[precision_weighting]]
      - [[policy_selection]]
---

# Active Inference

## Overview

Active Inference is a framework for understanding perception, learning, and decision-making based on the Free Energy Principle. It proposes that agents minimize expected free energy through both perception (inferring hidden states) and action (selecting policies that minimize expected surprise).

## Core Concepts

### Expected Free Energy
```math
G(π) = \sum_τ G(π,τ)
```
where:
- $G(π)$ is expected free energy for policy $π$
- $τ$ is future time point
- $G(π,τ)$ is expected free energy at time $τ$

### Policy Selection
```math
P(π) = σ(-γG(π))
```
where:
- $P(π)$ is policy probability
- $σ$ is softmax function
- $γ$ is precision parameter

## Mathematical Framework

### Discrete Time Active Inference

The discrete time formulation uses VFE for both perception and action:

```math
\begin{aligned}
& \text{Perception (State Inference):} \\
& F_t = \text{KL}[q(s_t)||p(s_t|o_{1:t})] - \ln p(o_t|o_{1:t-1}) \\
& \text{Action Selection:} \\
& a_t^* = \argmin_a \mathbb{E}_{q(s_t)}[F_{t+1}(a)] \\
& \text{Policy Selection:} \\
& P(\pi) = \sigma(-\gamma G(\pi)) \text{ where } G(\pi) = \sum_\tau F_{\tau}(\pi)
\end{aligned}
```

### Continuous Time Active Inference

The continuous time extension incorporates path integral formulation:

```math
\begin{aligned}
& \text{State Dynamics:} \\
& dF = \frac{\partial F}{\partial s}ds + \frac{1}{2}\text{tr}\left(\frac{\partial^2 F}{\partial s^2}D\right)dt \\
& \text{Action Selection:} \\
& a^* = \argmin_a \int_t^{t+dt} \mathcal{L}(s(\tau), \dot{s}(\tau), a) d\tau \\
& \text{Policy Selection:} \\
& P(\pi) = \sigma(-\gamma \int_t^{t+T} \mathcal{L}_\pi d\tau)
\end{aligned}
```

### Unified Framework

The relationship between discrete and continuous time is bridged by:

```math
\begin{aligned}
& F_{\text{discrete}} = -\ln p(o_t|s_t) - \ln p(s_t|s_{t-1}, a_{t-1}) + \ln q(s_t) \\
& F_{\text{continuous}} = \int_t^{t+dt} [\mathcal{L}(s,\dot{s},a) + \text{KL}[q(s(\tau))||p(s(\tau)|o(\tau))]] d\tau
\end{aligned}
```

## Implementation

### Active Inference Agent

```python
import numpy as np
from typing import List, Tuple, Optional
from scipy.special import softmax

class ActiveInferenceAgent:
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 n_actions: int,
                 learning_rate: float = 0.1,
                 temperature: float = 1.0):
        """Initialize active inference agent.
        
        Args:
            state_dim: State dimension
            obs_dim: Observation dimension
            n_actions: Number of actions
            learning_rate: Learning rate
            temperature: Action selection temperature
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr = learning_rate
        self.temp = temperature
        
        # Initialize model components
        self.init_model()
    
    def init_model(self):
        """Initialize model components."""
        # State-space model
        self.transition = np.random.randn(
            self.state_dim,
            self.state_dim,
            self.n_actions
        )
        self.emission = np.random.randn(
            self.obs_dim,
            self.state_dim
        )
        
        # Precision parameters
        self.state_precision = np.eye(self.state_dim)
        self.obs_precision = np.eye(self.obs_dim)
        
        # Beliefs
        self.state_belief = np.zeros(self.state_dim)
        self.policy_belief = np.ones(self.n_actions) / self.n_actions
    
    def infer_state(self,
                   observation: np.ndarray,
                   n_iterations: int = 10) -> np.ndarray:
        """Perform state inference.
        
        Args:
            observation: Observed data
            n_iterations: Number of iterations
            
        Returns:
            state: Inferred state
        """
        state = self.state_belief.copy()
        
        for _ in range(n_iterations):
            # Prediction error
            pred_obs = self.emission @ state
            obs_error = observation - pred_obs
            
            # State update
            state_grad = (
                self.emission.T @ self.obs_precision @ obs_error -
                self.state_precision @ state
            )
            
            state += self.lr * state_grad
        
        self.state_belief = state
        return state
    
    def compute_expected_free_energy(self,
                                   policy: np.ndarray) -> float:
        """Compute expected free energy for policy.
        
        Args:
            policy: Action policy
            
        Returns:
            G: Expected free energy
        """
        # Predicted states
        pred_states = []
        current_state = self.state_belief.copy()
        
        for action in policy:
            # State transition
            next_state = self.transition[:, :, action] @ current_state
            pred_states.append(next_state)
            current_state = next_state
        
        # Compute expected free energy components
        ambiguity = 0
        risk = 0
        
        for state in pred_states:
            # Predicted observations
            pred_obs = self.emission @ state
            
            # Ambiguity (entropy over observations)
            p_obs = softmax(pred_obs)
            ambiguity -= np.sum(p_obs * np.log(p_obs + 1e-10))
            
            # Risk (KL from preferred outcomes)
            risk += 0.5 * (
                state.T @ self.state_precision @ state +
                np.log(2 * np.pi / np.linalg.det(self.state_precision))
            )
        
        return ambiguity + risk
    
    def select_action(self,
                     policies: List[np.ndarray]) -> int:
        """Select action using active inference.
        
        Args:
            policies: List of possible policies
            
        Returns:
            action: Selected action
        """
        # Compute expected free energy
        G = np.array([
            self.compute_expected_free_energy(pi)
            for pi in policies
        ])
        
        # Policy selection
        self.policy_belief = softmax(-self.temp * G)
        
        # Sample action
        action = np.random.choice(
            len(policies),
            p=self.policy_belief
        )
        
        return action
```

### Environment Interface

```python
class Environment:
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 n_actions: int):
        """Initialize environment.
        
        Args:
            state_dim: State dimension
            obs_dim: Observation dimension
            n_actions: Number of actions
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        
        # Initialize dynamics
        self.init_dynamics()
    
    def init_dynamics(self):
        """Initialize environment dynamics."""
        # State transition model
        self.transition = np.random.randn(
            self.state_dim,
            self.state_dim,
            self.n_actions
        )
        
        # Observation model
        self.emission = np.random.randn(
            self.obs_dim,
            self.state_dim
        )
        
        # Current state
        self.state = np.random.randn(self.state_dim)
    
    def step(self,
            action: int) -> Tuple[np.ndarray, float]:
        """Take environment step.
        
        Args:
            action: Selected action
            
        Returns:
            observation: New observation
            reward: Reward value
        """
        # State transition
        self.state = self.transition[:, :, action] @ self.state
        
        # Generate observation
        observation = self.emission @ self.state
        
        # Compute reward
        reward = -np.sum(self.state**2)  # Example reward
        
        return observation, reward
```

### Training Loop

```python
def train_agent(agent: ActiveInferenceAgent,
               env: Environment,
               n_episodes: int = 100,
               episode_length: int = 50) -> List[float]:
    """Train active inference agent.
    
    Args:
        agent: Active inference agent
        env: Environment
        n_episodes: Number of episodes
        episode_length: Episode length
        
    Returns:
        rewards: Episode rewards
    """
    episode_rewards = []
    
    for episode in range(n_episodes):
        total_reward = 0
        observation, _ = env.step(0)  # Initial step
        
        for step in range(episode_length):
            # Generate policies
            policies = [
                np.random.randint(0, agent.n_actions, size=3)
                for _ in range(5)
            ]
            
            # Select action
            action = agent.select_action(policies)
            
            # Environment step
            next_obs, reward = env.step(action)
            
            # Update agent
            agent.infer_state(next_obs)
            
            # Update statistics
            total_reward += reward
            observation = next_obs
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode}, Reward: {total_reward:.2f}")
    
    return episode_rewards
```

## Best Practices

### Model Design
1. Choose appropriate dimensions
2. Initialize transition models
3. Set precision parameters
4. Design reward function

### Implementation
1. Monitor convergence
2. Handle numerical stability
3. Validate inference
4. Test policy selection

### Training
1. Tune learning rates
2. Adjust temperature
3. Balance exploration
4. Validate performance

## Common Issues

### Technical Challenges
1. State inference instability
2. Policy divergence
3. Reward sparsity
4. Local optima

### Solutions
1. Careful initialization
2. Gradient clipping
3. Reward shaping
4. Multiple restarts

## Related Documentation
- [[free_energy_principle]]
- [[predictive_coding]]
- [[policy_selection]]