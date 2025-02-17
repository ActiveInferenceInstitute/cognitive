---
type: visualization
id: vfe_components_001
created: 2024-02-05
modified: 2024-02-05
tags: [active-inference, free-energy, visualization]
aliases: [vfe-components, vfe-visualization]
---

# Variational Free Energy Components

## Overview

The Variational Free Energy (VFE) comprises two main components:
1. Accuracy Term (Negative Log-Likelihood)
2. Complexity Term (KL Divergence from Prior)

## Mathematical Formulation

$F = \underbrace{\mathbb{E}_{Q(x)}[-\ln P(y|x)]}_{\text{Accuracy}} + \underbrace{D_{KL}[Q(x)\|P(x)]}_{\text{Complexity}}$

## Component Analysis

### Accuracy Term
- Measures fit to observations
- Drives perceptual accuracy
- Computed as negative log-likelihood
- Links to [[likelihood_theory]]

### Complexity Term
- Measures deviation from prior
- Penalizes complex explanations
- Implements Occam's razor
- Links to [[information_complexity]]

## Visualization Implementation

```python
def plot_vfe_components_detailed(
    model,
    save: bool = True,
    output_dir: Optional[Path] = None
) -> plt.Figure:
    """Plot detailed breakdown of Variational Free Energy components.
    
    Args:
        model: Active Inference model instance
        save: Whether to save the plot
        output_dir: Output directory for saving
        
    Returns:
        Matplotlib figure with multiple subplots
    """
    # Create figure with GridSpec
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # Get history data
    accuracy = np.array(model.state.history['accuracy'])
    complexity = np.array(model.state.history['complexity'])
    total_vfe = accuracy + complexity
    time_steps = np.arange(len(accuracy))
    
    # 1. Total VFE Plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_steps, total_vfe, 'k-', label='Total VFE')
    ax1.set_title('Total Variational Free Energy')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.grid(True)
    
    # Add trend line
    z = np.polyfit(time_steps, total_vfe, 1)
    p = np.poly1d(z)
    ax1.plot(time_steps, p(time_steps), 'r--', label='Trend')
    ax1.legend()
    
    # 2. Components Stacked Area Plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(time_steps, 0, accuracy, alpha=0.5,
                    label='Accuracy', color='purple')
    ax2.fill_between(time_steps, accuracy, accuracy + complexity,
                    alpha=0.5, label='Complexity', color='orange')
    ax2.set_title('VFE Components (Stacked)')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Component Ratio Plot
    ax3 = fig.add_subplot(gs[1, 0])
    ratio = np.abs(accuracy) / (np.abs(accuracy) + np.abs(complexity) + 1e-10)
    ax3.plot(time_steps, ratio, 'purple', label='Accuracy Ratio')
    ax3.plot(time_steps, 1 - ratio, 'orange', label='Complexity Ratio')
    ax3.set_title('Component Ratio')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Ratio')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Component Scatter Plot
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(accuracy, complexity, c=time_steps, cmap='viridis')
    ax4.set_title('Accuracy vs Complexity')
    ax4.set_xlabel('Accuracy Term')
    ax4.set_ylabel('Complexity Term')
    plt.colorbar(scatter, ax=ax4, label='Time Step')
    ax4.grid(True)
    
    # 5. Running Averages
    ax5 = fig.add_subplot(gs[2, :])
    window = min(5, len(time_steps))
    if window > 1:
        running_avg_accuracy = np.convolve(accuracy,
                                         np.ones(window)/window,
                                         mode='valid')
        running_avg_complexity = np.convolve(complexity,
                                           np.ones(window)/window,
                                           mode='valid')
        running_avg_total = np.convolve(total_vfe,
                                      np.ones(window)/window,
                                      mode='valid')
        valid_steps = time_steps[window-1:]
        
        ax5.plot(valid_steps, running_avg_accuracy,
                'purple', label='Accuracy (Avg)')
        ax5.plot(valid_steps, running_avg_complexity,
                'orange', label='Complexity (Avg)')
        ax5.plot(valid_steps, running_avg_total,
                'k-', label='Total VFE (Avg)')
    ax5.set_title(f'Running Averages (Window={window})')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Value')
    ax5.legend()
    ax5.grid(True)
    
    plt.tight_layout()
    
    if save and output_dir:
        plt.savefig(output_dir / 'vfe_components_detailed.png')
    
    return fig
```

## Plot Descriptions

### 1. Total VFE Plot
- Shows overall Variational Free Energy over time
- Includes trend line for convergence analysis
- Links to [[free_energy_minimization]]

### 2. Components Stacked Area
- Purple area: Accuracy term (data fit)
- Orange area: Complexity term (prior deviation)
- Shows relative contribution of each
- Links to [[model_complexity]]

### 3. Component Ratio
- Shows balance between accuracy/complexity
- Purple line: Proportion of accuracy term
- Orange line: Proportion of complexity term
- Links to [[model_selection]]

### 4. Component Scatter
- X-axis: Accuracy term
- Y-axis: Complexity term
- Color: Time progression
- Links to [[learning_trajectory]]

### 5. Running Averages
- Smoothed trends of both components
- Helps identify learning phases
- Links to [[convergence_analysis]]

## Analysis Methods

### Temporal Patterns
- Initial high complexity phase
- Convergence to optimal balance
- Learning rate effects
- Links to [[learning_dynamics]]

### Balance Analysis
- Trade-off between fit and complexity
- Model selection implications
- Optimization dynamics
- Links to [[optimization_theory]]

### Performance Metrics
- Average total VFE
- Component balance
- Convergence rate
- Links to [[performance_analysis]]

## Related Visualizations
- [[belief_evolution]]
- [[model_comparison]]
- [[free_energy_landscape]]

## References
- [[friston_2006]] - Free Energy Principle
- [[bogacz_2017]] - Tutorial on VFE
- [[buckley_2017]] - Model Selection 