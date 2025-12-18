---

title: Visualization Tools

type: concept

status: stable

tags:
  - visualization
  - tools
  - analysis

semantic_relations:
  - type: relates
    links:
      - graph_visualization
      - matrix_plots
      - diagnostics
      - network_analysis
      - performance_optimization
      - model_validation
  - type: implements
    links:
      - [[../mathematics/matrix_operations]]
      - [[../mathematics/probability_distributions]]
  - type: foundation
    links:
      - [[../mathematics/visualization_mathematics]]
      - [[information_processing]]

---

# Visualization Tools

Visualization tools are essential for understanding, debugging, and communicating cognitive models. They provide intuitive representations of complex probabilistic processes, model dynamics, and inference results in active inference and partially observable Markov decision processes (POMDPs).

## Core Visualization Types

### Generative Model Components

#### Likelihood Matrices (A)

```python
def plot_likelihood_matrix(A_matrix, observation_names, state_names):
    """Visualize sensory likelihood mappings."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(A_matrix, xticklabels=state_names, yticklabels=observation_names,
                cmap='viridis', annot=True, fmt='.2f')
    plt.title('Likelihood Matrix A (p(o|s))')
    plt.xlabel('Hidden States')
    plt.ylabel('Observations')
    plt.tight_layout()
    plt.show()
```

**Key Features:**
- Probability distributions over observations for each state
- State-observation mapping clarity
- Likelihood concentration analysis

#### Transition Matrices (B)

```python
def plot_transition_matrix(B_matrix, action_names, state_names):
    """Visualize state transition dynamics."""
    n_actions, n_states, _ = B_matrix.shape

    fig, axes = plt.subplots(1, n_actions, figsize=(15, 5))
    for i, ax in enumerate(axes):
        sns.heatmap(B_matrix[i], xticklabels=state_names, yticklabels=state_names,
                    cmap='plasma', annot=True, fmt='.2f', ax=ax)
        ax.set_title(f'Transition Matrix B (Action: {action_names[i]})')
        ax.set_xlabel('Next State')
        ax.set_ylabel('Current State')
    plt.tight_layout()
    plt.show()
```

**Key Features:**
- Action-dependent state transitions
- Temporal dynamics visualization
- Markov chain structure analysis

#### Prior Preferences (C)

```python
def plot_prior_preferences(C_matrix, observation_names):
    """Visualize preferred observation distributions."""
    plt.figure(figsize=(10, 6))
    for i, obs in enumerate(observation_names):
        plt.plot(C_matrix[i], label=obs, marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Log Probability')
    plt.title('Prior Preferences C')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

**Key Features:**
- Goal-directed behavior encoding
- Temporal preference structure
- Homeostatic drive visualization

### Inference Results

#### Belief Trajectories

```python
def plot_belief_trajectories(belief_history, state_names, uncertainty_bands=True):
    """Visualize belief evolution over time."""
    plt.figure(figsize=(12, 8))

    for i, state in enumerate(state_names):
        beliefs = [b[i] for b in belief_history]
        plt.plot(beliefs, label=state, linewidth=2)

        if uncertainty_bands:
            # Compute uncertainty (1 - max_belief) or use variance if available
            uncertainty = [1 - max(b) for b in belief_history]
            plt.fill_between(range(len(beliefs)),
                           [b - u/2 for b, u in zip(beliefs, uncertainty)],
                           [b + u/2 for b, u in zip(beliefs, uncertainty)],
                           alpha=0.3)

    plt.xlabel('Time Step')
    plt.ylabel('Belief Probability')
    plt.title('Belief Trajectories with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

**Key Features:**
- Temporal belief evolution
- Uncertainty quantification
- State estimation accuracy

#### Expected Free Energy (EFE) Components

```python
def plot_efe_components(efe_components, action_names):
    """Visualize components of expected free energy."""
    components = ['Epistemic', 'Extrinsic', 'Intrinsic']

    fig, axes = plt.subplots(1, len(components), figsize=(15, 5))

    for i, (component, ax) in enumerate(zip(components, axes)):
        component_values = [efe_comp[i] for efe_comp in efe_components]
        ax.bar(action_names, component_values)
        ax.set_title(f'{component} Value')
        ax.set_ylabel('EFE Component')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
```

**Key Features:**
- Action value decomposition
- Exploration vs exploitation balance
- Information-seeking behavior

### Policy and Action Selection

#### Policy Posterior Visualization

```python
def plot_policy_posteriors(policy_posteriors, policy_descriptions):
    """Visualize policy probability distributions."""
    plt.figure(figsize=(10, 6))

    for i, policy in enumerate(policy_descriptions):
        posterior = [p[i] for p in policy_posteriors]
        plt.plot(posterior, label=policy, marker='o')

    plt.xlabel('Time Step')
    plt.ylabel('Policy Probability')
    plt.title('Policy Posterior Evolution')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

**Key Features:**
- Policy competition dynamics
- Decision-making processes
- Action selection probability

#### Phase Space Analysis

```python
def plot_phase_space(belief_trajectory, action_trajectory, state_names):
    """Visualize system dynamics in phase space."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(belief_trajectory)))

    for i in range(len(belief_trajectory)-1):
        ax.quiver(belief_trajectory[i][0], belief_trajectory[i][1], belief_trajectory[i][2],
                 belief_trajectory[i+1][0] - belief_trajectory[i][0],
                 belief_trajectory[i+1][1] - belief_trajectory[i][1],
                 belief_trajectory[i+1][2] - belief_trajectory[i][2],
                 color=colors[i], alpha=0.6)

    ax.set_xlabel(f'Belief in {state_names[0]}')
    ax.set_ylabel(f'Belief in {state_names[1]}')
    ax.set_zlabel(f'Belief in {state_names[2]}')
    ax.set_title('Phase Space Trajectory')
    plt.show()
```

**Key Features:**
- System attractor analysis
- Dynamic stability assessment
- State space exploration

## Advanced Visualization Techniques

### Interactive Dashboards

```python
def create_interactive_dashboard(model, belief_history, action_history):
    """Create interactive visualization dashboard."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Belief Trajectories', 'Policy Selection',
                       'EFE Components', 'Phase Space'),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'scatter3d'}]]
    )

    # Add belief trajectories
    for i, state in enumerate(model.state_names):
        beliefs = [b[i] for b in belief_history]
        fig.add_trace(go.Scatter(x=list(range(len(beliefs))), y=beliefs,
                                mode='lines', name=f'Belief {state}'),
                     row=1, col=1)

    # Add policy selection
    policy_probs = [p[0] for p in model.policy_posteriors]  # Example for first policy
    fig.add_trace(go.Bar(x=model.policy_names, y=policy_probs),
                 row=1, col=2)

    # Add EFE components
    efe_epistemic = [efe[0] for efe in model.expected_free_energies]
    efe_extrinsic = [efe[1] for efe in model.expected_free_energies]
    fig.add_trace(go.Scatter(x=list(range(len(efe_epistemic))), y=efe_epistemic,
                            mode='lines', name='Epistemic EFE'),
                 row=2, col=1)

    # Add 3D phase space
    beliefs_x = [b[0] for b in belief_history]
    beliefs_y = [b[1] for b in belief_history]
    beliefs_z = [b[2] for b in belief_history]
    fig.add_trace(go.Scatter3d(x=beliefs_x, y=beliefs_y, z=beliefs_z,
                              mode='lines', name='Trajectory'),
                 row=2, col=2)

    fig.update_layout(height=800, title_text="Active Inference Model Dashboard")
    fig.show()
```

### Real-time Monitoring

```python
class RealTimeVisualizer:
    """Real-time visualization during model execution."""

    def __init__(self, model, update_interval=0.1):
        self.model = model
        self.update_interval = update_interval
        self.setup_plots()

    def setup_plots(self):
        """Initialize real-time plotting."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.belief_lines = []
        self.policy_bars = None

        # Setup belief trajectory plot
        self.axes[0,0].set_title('Real-time Beliefs')
        self.axes[0,0].set_xlabel('Time')
        self.axes[0,0].set_ylabel('Probability')

    def update_visualization(self, current_beliefs, current_policies):
        """Update plots with new data."""
        # Update belief trajectories
        for i, line in enumerate(self.belief_lines):
            line.set_ydata(np.append(line.get_ydata(), current_beliefs[i]))

        # Update policy bars
        if self.policy_bars:
            for bar, height in zip(self.policy_bars, current_policies):
                bar.set_height(height)

        plt.pause(self.update_interval)
```

## Diagnostic Visualizations

### Model Validation Plots

```python
def plot_model_validation(true_states, predicted_beliefs, observations):
    """Validate model predictions against ground truth."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # True vs predicted states
    axes[0].plot(true_states, label='True State', linewidth=2)
    predicted_states = [np.argmax(b) for b in predicted_beliefs]
    axes[0].plot(predicted_states, label='Predicted State', linestyle='--')
    axes[0].set_title('State Estimation Accuracy')
    axes[0].legend()

    # Prediction confidence
    confidence = [max(b) for b in predicted_beliefs]
    axes[1].plot(confidence, color='orange')
    axes[1].set_title('Prediction Confidence')
    axes[1].set_ylabel('Max Belief Probability')

    # Observation likelihood
    obs_likelihood = []
    for t, obs in enumerate(observations):
        likelihood = predicted_beliefs[t][true_states[t]]
        obs_likelihood.append(likelihood)
    axes[2].plot(obs_likelihood, color='green')
    axes[2].set_title('Observation Likelihood Under True State')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Likelihood')

    plt.tight_layout()
    plt.show()
```

### Performance Metrics Dashboard

```python
def plot_performance_metrics(metrics_history):
    """Visualize model performance over time."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'EFE']

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12))

    for i, metric in enumerate(metrics):
        if metric in metrics_history:
            values = metrics_history[metric]
            axes[i].plot(values, marker='o')
            axes[i].set_title(f'{metric} Over Time')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()
```

## Best Practices

### Design Principles

- **Clarity**: Use appropriate color schemes and annotations
- **Interactivity**: Enable zooming, panning, and data exploration
- **Consistency**: Maintain visual conventions across related plots
- **Accessibility**: Ensure colorblind-friendly palettes and clear labels

### Implementation Guidelines

- **Modularity**: Separate visualization logic from model computation
- **Efficiency**: Use appropriate data structures for real-time updates
- **Reproducibility**: Include code for regenerating figures
- **Documentation**: Explain the meaning and interpretation of each visualization

## Integration with Analysis Tools

### Export Formats

- **Static Images**: PNG, SVG, PDF for publications
- **Interactive HTML**: Plotly, Bokeh for web-based exploration
- **Video Sequences**: MP4, GIF for dynamic process visualization
- **Data Files**: CSV, JSON for further analysis

### Tool Integration

```python
# Integration with analysis pipeline
def visualization_pipeline(model_results, config):
    """Complete visualization workflow."""

    # Generate core visualizations
    plot_belief_trajectories(model_results.beliefs)
    plot_efe_components(model_results.efe_history)
    plot_policy_posteriors(model_results.policy_history)

    # Generate diagnostics
    plot_model_validation(model_results.true_states,
                         model_results.beliefs,
                         model_results.observations)

    # Export results
    export_visualizations(config.output_dir, config.formats)

    return visualization_report
```

---

## Related Concepts

- [[graph_visualization]] - Graph-based representation techniques
- [[matrix_plots]] - Matrix visualization methods
- [[diagnostics]] - Model diagnostic procedures
- [[performance_optimization]] - Optimization visualization
- [[model_validation]] - Validation techniques

