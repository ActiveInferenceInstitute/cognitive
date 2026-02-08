---
title: Visualization Suite
type: concept
status: stable
created: 2026-02-06
updated: 2026-02-07
tags:
  - visualization
  - plotting
  - diagnostics
  - active-inference
  - analysis
semantic_relations:
  - type: relates
    links:
      - [[graph_visualization]]
      - [[visualization_tools]]
      - [[performance_metrics]]
      - [[simulation_studies]]
---

# Visualization Suite

## Overview

The visualization suite provides standardized plotting and diagnostic tools for Active Inference agents. Visualizations span four categories: belief dynamics, free energy landscapes, policy evaluation, and learning trajectories, enabling researchers to diagnose inference quality and communicate results.

## Belief Dynamics Visualization

### State Belief Evolution

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_belief_evolution(beliefs_history, state_labels=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    beliefs = np.array(beliefs_history)
    n_states = beliefs.shape[1]
    
    for i in range(n_states):
        label = state_labels[i] if state_labels else f"State {i}"
        ax.plot(beliefs[:, i], label=label, linewidth=2)
    
    ax.set_xlabel('Time step', fontsize=14)
    ax.set_ylabel('Belief probability', fontsize=14)
    ax.set_title('State Belief Evolution', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    return ax
```

### Free Energy Landscape

```python
def plot_free_energy_landscape(free_energies, policies, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(policies)))
    bars = ax.bar(range(len(policies)), free_energies, color=colors, edgecolor='black')
    
    ax.set_xlabel('Policy', fontsize=14)
    ax.set_ylabel('Expected Free Energy G(π)', fontsize=14)
    ax.set_title('Policy Evaluation', fontsize=16)
    ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
    return ax
```

## Diagnostic Plots

| Plot Type | Purpose | Data Source |
| --- | --- | --- |
| Belief heatmap | State beliefs over time across factors | $q(s_t)$ |
| EFE bar chart | Compare policies | $G(\pi)$ |
| Prediction error trace | Track inference quality | $\varepsilon_t$ |
| Learning curve | Parameter convergence | $A_t, B_t$ |
| Free energy trace | Overall model performance | $F_t$ |
| Precision dynamics | Attention/confidence | $\gamma_t$ |
| Phase portrait | Continuous dynamics | $\mu_t$ |

## Multi-Panel Dashboard

```python
def create_agent_dashboard(agent_history):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    plot_belief_evolution(agent_history['beliefs'], ax=axes[0, 0])
    plot_free_energy_landscape(agent_history['G'], agent_history['policies'], ax=axes[0, 1])
    axes[0, 2].plot(agent_history['free_energy'], 'b-', linewidth=2)
    axes[0, 2].set_title('Free Energy', fontsize=16)
    
    axes[1, 0].plot(agent_history['prediction_errors'], 'r-', alpha=0.7)
    axes[1, 0].set_title('Prediction Errors', fontsize=16)
    axes[1, 1].plot(agent_history['precision'], 'g-', linewidth=2)
    axes[1, 1].set_title('Precision γ', fontsize=16)
    axes[1, 2].plot(agent_history['reward'], 'k-', linewidth=2)
    axes[1, 2].set_title('Cumulative Reward', fontsize=16)
    
    plt.tight_layout()
    return fig
```

## Related Topics

- [[graph_visualization]] — Graph-based visualization
- [[visualization_tools]] — Visualization tooling
- [[performance_metrics]] — Performance metrics
- [[simulation_studies]] — Simulation analysis
