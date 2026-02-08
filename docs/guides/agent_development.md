---
title: Agent Development Guide
type: guide
status: stable
created: 2025-01-01
updated: 2026-02-07
tags:
  - agents
  - development
  - active_inference
  - implementation
semantic_relations:
  - type: relates_to
    links:
      - [[docs/implementation/implementation_guides]]
      - [[code/Things/Simple_POMDP/README]]
      - [[knowledge_base/agents/README]]
---

# Agent Development Guide

## Overview

This guide covers the development of cognitive agents using the Active Inference framework within the Cognitive repository. It walks through agent architecture, belief updating, policy selection, and integration with the POMDP environment.

## Core Agent Architecture

An Active Inference agent maintains a **generative model** of its environment and uses variational inference to update beliefs and select actions that minimize expected free energy.

### Key Components

1. **Generative Model** — encodes the agent's assumptions about how hidden states produce observations and how states transition
2. **Belief State** — a probability distribution over hidden states, updated via variational message passing
3. **Policy Selection** — actions chosen by evaluating the expected free energy of candidate policies
4. **Learning** — model parameters updated over time through experience

### Minimal Agent Structure

```python
class ActiveInferenceAgent:
    """Minimal Active Inference agent following the POMDP formulation."""

    def __init__(self, A, B, C, D, policy_len=1):
        """
        Args:
            A: Observation likelihood matrix (what I expect to see given a state)
            B: Transition matrix (how states change given actions)
            C: Preference vector (desired observations)
            D: Prior over initial states
            policy_len: Planning horizon
        """
        self.A = A  # Likelihood
        self.B = B  # Transition
        self.C = C  # Preferences
        self.D = D  # Prior
        self.qs = D.copy()  # Current beliefs
        self.policy_len = policy_len

    def infer_states(self, observation):
        """Update beliefs given a new observation using Bayesian inference."""
        likelihood = self.A[observation, :]
        self.qs = likelihood * self.qs
        self.qs /= self.qs.sum()
        return self.qs

    def select_action(self):
        """Select action minimizing expected free energy."""
        G = []
        for action in range(self.B.shape[2]):
            qs_next = self.B[:, :, action] @ self.qs
            expected_obs = self.A @ qs_next
            # Epistemic value (information gain) + pragmatic value (preference alignment)
            G_action = expected_obs @ (np.log(expected_obs + 1e-16) - np.log(self.C + 1e-16))
            G.append(G_action)
        return np.argmin(G)
```

## Development Workflow

### Step 1: Define the Generative Model

Start by specifying the POMDP matrices:

- **A** (likelihood): Maps hidden states to observations
- **B** (transition): Maps state-action pairs to next states
- **C** (preferences): Encodes the agent's goals as preferred observations
- **D** (prior): Initial belief over states

### Step 2: Implement Belief Updating

Use variational message passing or fixed-point iteration to update the agent's posterior beliefs about hidden states given observations.

### Step 3: Implement Policy Selection

Evaluate candidate policies by computing the **expected free energy** (EFE), which balances:

- **Epistemic value**: seeking information to reduce uncertainty
- **Pragmatic value**: seeking preferred observations

### Step 4: Run the Agent-Environment Loop

```python
for t in range(num_steps):
    observation = environment.observe()
    agent.infer_states(observation)
    action = agent.select_action()
    environment.step(action)
```

### Step 5: Evaluate and Iterate

Measure agent performance against domain-specific metrics and refine the generative model accordingly.

## Reference Implementations

| Implementation | Location | Description |
|---|---|---|
| Simple POMDP | [[code/Things/Simple_POMDP/README]] | Minimal discrete-state POMDP agent |
| Generic POMDP | [[code/Things/Generic_POMDP/AGENTS]] | Configurable POMDP agent framework |
| Ant Colony | [[code/Things/Ant_Colony/AGENTS]] | Multi-agent swarm intelligence |
| BioFirm | `code/Things/BioFirm/` | Organizational Active Inference |

## Best Practices

- **Start simple**: Begin with small state/observation spaces and expand incrementally
- **Validate matrices**: Ensure A and B are proper stochastic matrices (columns/rows sum to 1)
- **Log everything**: Record beliefs, actions, and free energy at each timestep for debugging
- **Test with known solutions**: Verify agent behavior on problems with known optimal policies
- **Use typed configurations**: Leverage Python dataclasses or Pydantic for model parameters

## Related Resources

- [[docs/implementation/implementation_guides|Implementation Guides]] — detailed implementation patterns
- [[knowledge_base/cognitive/active_inference|Active Inference Theory]] — theoretical foundations
- [[knowledge_base/mathematics/pomdp_framework|POMDP Framework]] — mathematical formulation
- [[docs/guides/best_practices|Best Practices]] — coding and design best practices
- [[docs/guides/AGENTS|Agent Guide Index]] — comprehensive agent development reference
