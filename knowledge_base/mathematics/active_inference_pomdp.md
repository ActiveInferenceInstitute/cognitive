---
type: mathematical_concept
id: active_inference_pomdp_001
created: 2024-02-05
modified: 2024-02-06
tags: [active-inference, pomdp, free-energy, mathematics]
aliases: [active-inference-pomdp]
semantic_relations:
  - type: implements
    links: 
      - [[../../docs/research/research_documentation_index|Research Documentation]]
      - [[free_energy_principle]]
  - type: uses
    links:
      - [[variational_methods]]
      - [[information_theory]]
  - type: documented_by
    links:
      - [[../../docs/guides/implementation_guides_index|Implementation Guides]]
      - [[../../docs/api/api_documentation_index|API Documentation]]
---

# Active Inference POMDP

## Overview

Active Inference formulation of Partially Observable Markov Decision Processes (POMDPs) combines:
- [[variational_free_energy]] for perception (sensemaking)
- [[expected_free_energy]] for action (decision-making)
- [[belief_updating]] for state inference
- [[policy_selection]] for action selection

## Core Components

### State Space
- Hidden states $s \in S$
- Observations $o \in O$
- Actions $a \in A$
- Policies $\pi \in \Pi$

### Generative Model
- [[A_matrix]]: Observation model $P(o|s)$
- [[B_matrix]]: Transition model $P(s'|s,a)$
- [[C_matrix]]: Preferences $P(o)$
- [[D_matrix]]: Prior beliefs $P(s_1)$
- [[E_matrix]]: Action distribution $P(a)$

## Matrix Properties and Validation

### Observation Model (A)
- Column stochastic: $\sum_o P(o|s) = 1$
- Non-negative: $P(o|s) \geq 0$
- Links to [[likelihood_mapping]] and [[sensory_uncertainty]]

### Transition Model (B)
- Column stochastic per action: $\sum_{s'} P(s'|s,a) = 1$
- Non-negative: $P(s'|s,a) \geq 0$
- Links to [[state_dynamics]] and [[action_effects]]

### Preference Model (C)
- Log-probability format: $C = \ln P(o)$
- Finite values: $C \in \mathbb{R}$
- Links to [[utility_theory]] and [[goal_specification]]

### Prior Beliefs (D)
- Normalized: $\sum_s P(s) = 1$
- Non-negative: $P(s) \geq 0$
- Links to [[belief_initialization]] and [[prior_knowledge]]

### Action Distribution (E)
- Normalized: $\sum_a P(a) = 1$
- Non-negative: $P(a) \geq 0$
- Links to [[action_selection]] and [[policy_distribution]]

## Free Energy Formulations

### Variational Free Energy (VFE)
$F = \mathbb{E}_{Q(s)}[\ln Q(s) - \ln P(o,s)]$

Components:
- Accuracy: $-\mathbb{E}_{Q(s)}[\ln P(o|s)]$
- Complexity: $D_{KL}[Q(s)\|P(s)]$

### Expected Free Energy (EFE)
$G(\pi) = \underbrace{-\mathbb{E}_{Q(\tilde{x},\tilde{y}|\pi)}[D_{KL}[Q(\tilde{x}|\tilde{y},\pi)\|Q(\tilde{x}|\pi)]]}_{\text{Information gain}} - \underbrace{\mathbb{E}_{Q(\tilde{y}|\pi)}[\ln P(\tilde{y}|C)]}_{\text{Pragmatic value}}$

Alternative formulations:
```latex
\begin{aligned}
G(\pi) &= \underbrace{\mathbb{E}_{Q(\tilde{x}|\pi)}[H[P(\tilde{y}|\tilde{x})]]}_{\text{Expected ambiguity}} + \underbrace{D_{KL}[Q(\tilde{y}|\pi)\|P(\tilde{y}|\pi)]}_{\text{Risk}} \\
&= \underbrace{-\mathbb{E}_{Q(\tilde{x},\tilde{y}|\pi)}[\ln P(\tilde{y},\tilde{x}|C)]}_{\text{Expected energy}} - \underbrace{H[Q(\tilde{x}|\pi)]}_{\text{Entropy}}
\end{aligned}
```

## Implementation

### Core Methods
- [[compute_vfe]]: Calculate Variational Free Energy
- [[compute_efe]]: Calculate Expected Free Energy
- [[update_beliefs]]: Belief updating using VFE
- [[select_policy]]: Policy selection using EFE

### Matrix Validation
```python
def validate_matrices(A, B, C, D, E):
    """Validate matrix properties."""
    # A matrix validation
    assert np.allclose(A.sum(axis=0), 1.0), "A matrix not column stochastic"
    assert np.all(A >= 0), "A matrix has negative values"
    
    # B matrix validation
    for a in range(B.shape[2]):
        assert np.allclose(B[:,:,a].sum(axis=0), 1.0), f"B matrix not column stochastic for action {a}"
        assert np.all(B[:,:,a] >= 0), f"B matrix has negative values for action {a}"
    
    # D matrix validation
    assert np.allclose(D.sum(), 1.0), "D matrix not normalized"
    assert np.all(D >= 0), "D matrix has negative values"
    
    # E matrix validation
    assert np.allclose(E.sum(), 1.0), "E matrix not normalized"
    assert np.all(E >= 0), "E matrix has negative values"
```

### Key Properties
- VFE bounds surprise: $F \geq -\ln P(o)$
- EFE balances exploration and exploitation
- Policy selection uses softmax: $P(\pi) = \sigma(-\gamma G(\pi))$

## Integration

### Active Inference Loop
1. State Estimation ([[belief_updating]])
   ```python
   def update_state_estimate(observation, action):
       """Update belief state using new observation."""
       likelihood = compute_likelihood(observation)  # Using A matrix
       transition = predict_state(action)           # Using B matrix
       return combine_evidence(likelihood, transition)
   ```

2. Policy Selection ([[policy_selection]])
   ```python
   def select_action(beliefs, temperature=1.0):
       """Select action using Expected Free Energy."""
       G = compute_expected_free_energy(beliefs)    # Using C matrix
       E = softmax(-temperature * G)               # Update E matrix
       return sample_action(E)
   ```

3. Learning ([[parameter_learning]])
   ```python
   def update_parameters(observation, action, reward):
       """Update model parameters based on experience."""
       update_observation_model(observation)        # Update A matrix
       update_transition_model(action)             # Update B matrix
       update_preferences(reward)                  # Update C matrix
   ```

## Visualization

### Key Plots
- [[belief_evolution]]: State inference over time
- [[free_energy_landscape]]: VFE/EFE surfaces
- [[efe_components]]: Epistemic vs Pragmatic value

### Matrix Visualization
```python
def plot_matrices(A, B, C, D, E):
    """Visualize all matrices."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # A matrix (top-left)
    ax_a = fig.add_subplot(gs[0, 0])
    sns.heatmap(A, ax=ax_a, cmap='YlOrRd')
    ax_a.set_title('A: Observation Model')
    
    # B matrix (top-middle)
    ax_b = fig.add_subplot(gs[0, 1])
    for a in range(B.shape[2]):
        plt.subplot(1, B.shape[2], a+1)
        sns.heatmap(B[:,:,a])
        plt.title(f'B: Transitions (Action {a})')
    
    # C matrix (top-right)
    ax_c = fig.add_subplot(gs[0, 2])
    sns.barplot(x=range(len(C)), y=C)
    ax_c.set_title('C: Preferences')
    
    # D matrix (middle)
    ax_d = fig.add_subplot(gs[1, :])
    sns.barplot(x=range(len(D)), y=D)
    ax_d.set_title('D: Prior Beliefs')
    
    # E matrix (bottom)
    ax_e = fig.add_subplot(gs[2, :])
    sns.barplot(x=range(len(E)), y=E)
    ax_e.set_title('E: Action Distribution')
```

## Related Concepts
- [[active_inference_theory]]
- [[free_energy_principle]]
- [[markov_decision_process]]
- [[variational_inference]]
- [[information_geometry]]
- [[belief_propagation]]
- [[optimal_control]]

## Documentation Links
- [[../../docs/research/research_documentation_index|Research Documentation]]
- [[../../docs/guides/implementation_guides_index|Implementation Guides]]
- [[../../docs/api/api_documentation_index|API Documentation]]
- [[../../docs/examples/usage_examples_index|Usage Examples]]

## References
- [[friston_2017]] - Active Inference
- [[da_costa_2020]] - Active Inference POMDP
- [[parr_2019]] - Generalizing Free Energy
- [[buckley_2017]] - Free Energy Tutorial 