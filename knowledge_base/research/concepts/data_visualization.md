---
title: Data Visualization
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [visualization, plotting, analysis, data_science]
semantic_relations:
  - type: relates
    links: [[[policy_visualization]], [[knowledge_base/cognitive/visualization_suite]], [[knowledge_base/cognitive/graph_visualization]]]
---

# Data Visualization

Techniques for visualizing Active Inference simulation data, including time series, phase portraits, information-theoretic diagrams, and comparative analyses.

## Standard Plot Types

### Time Series

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_results(results, title="Active Inference Simulation"):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(results['observations'], 'b-', alpha=0.7)
    axes[0].set_ylabel("Observations")
    axes[1].imshow(results['beliefs'].T, aspect='auto', cmap='viridis')
    axes[1].set_ylabel("States")
    axes[2].plot(results['free_energy'], 'r-')
    axes[2].set_ylabel("Free Energy")
    axes[3].plot(results['actions'], 'g-', drawstyle='steps')
    axes[3].set_ylabel("Actions")
    axes[3].set_xlabel("Time Step")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig
```

### Phase Portraits

```python
def plot_phase_portrait(states, dims=(0, 1)):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(states[:, dims[0]], states[:, dims[1]], 'b-', alpha=0.5)
    ax.scatter(states[0, dims[0]], states[0, dims[1]], c='green', s=100, marker='o', label='Start')
    ax.scatter(states[-1, dims[0]], states[-1, dims[1]], c='red', s=100, marker='x', label='End')
    ax.set_xlabel(f"State dim {dims[0]}")
    ax.set_ylabel(f"State dim {dims[1]}")
    ax.legend()
    return fig
```

### Information Diagrams

```mermaid
graph TD
    subgraph "Information Decomposition"
        HO[H(O)] --> MI[I(O;S)]
        HS[H(S)] --> MI
        MI --> HOS[H(O|S)]
        MI --> HSO[H(S|O)]
    end
    style MI fill:#bbf,stroke:#333
```

## Related Topics

- [[policy_visualization]] — Policy-specific visualizations
- [[knowledge_base/cognitive/visualization_suite]] — Comprehensive visualization tools
- [[knowledge_base/cognitive/graph_visualization]] — Graph visualizations
