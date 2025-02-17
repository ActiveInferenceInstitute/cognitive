---
type: implementation
id: compute_efe_001
created: 2024-02-05
modified: 2024-02-05
tags: [active-inference, free-energy, implementation]
aliases: [compute-efe, expected-free-energy]
---

# Computing Expected Free Energy

## Mathematical Definition

The Expected Free Energy (EFE) for a policy $\pi$ is:

$G(\pi) = \underbrace{-\mathbb{E}_{Q(\tilde{x},\tilde{y}|\pi)}[D_{KL}[Q(\tilde{x}|\tilde{y},\pi)\|Q(\tilde{x}|\pi)]]}_{\text{Epistemic Value}} - \underbrace{\mathbb{E}_{Q(\tilde{y}|\pi)}[\ln P(\tilde{y}|C)]}_{\text{Pragmatic Value}}$

## Implementation

```python
def compute_expected_free_energy(
    A: np.ndarray,           # Observation model P(o|s)
    B: np.ndarray,           # Transition model P(s'|s,a)
    C: np.ndarray,           # Log preferences ln P(o)
    beliefs: np.ndarray,     # Current state beliefs Q(s)
    action: int             # Action to evaluate
) -> Tuple[float, float, float]:
    """Compute Expected Free Energy for a single action.
    
    Args:
        A: Observation likelihood matrix [n_obs x n_states]
        B: State transition tensor [n_states x n_states x n_actions]
        C: Log preference vector [n_obs]
        beliefs: Current belief state [n_states]
        action: Action index to evaluate
        
    Returns:
        Tuple of (total_EFE, epistemic_value, pragmatic_value) where:
        - total_EFE: Total Expected Free Energy
        - epistemic_value: Information gain (uncertainty reduction)
        - pragmatic_value: Preference satisfaction (utility)
    """
    # Predicted next state distribution
    Qs_a = B[:, :, action] @ beliefs
    
    # Predicted observation distribution
    Qo_a = A @ Qs_a
    
    # Epistemic value (state uncertainty/information gain)
    epistemic = compute_entropy(Qs_a)
    
    # Pragmatic value (preference satisfaction/utility)
    pragmatic = -np.sum(Qo_a * C)  # Negative because C is log preferences
    
    # Total Expected Free Energy
    total_efe = epistemic + pragmatic
    
    return total_efe, epistemic, pragmatic
```

## Components

### Epistemic Value
- Information gain about hidden states
- Drives exploration and uncertainty reduction
- Computed using entropy of predicted state distribution
- Links to [[information_theory]]

### Pragmatic Value
- Goal-directed behavior
- Drives exploitation of preferences
- Computed using KL divergence from preferred outcomes
- Links to [[utility_theory]]

## Usage

### In Policy Selection
```python
def select_policy(model, temperature: float = 1.0) -> int:
    """Select action using Expected Free Energy."""
    G = np.zeros(model.num_actions)
    
    for a in range(model.num_actions):
        G[a], _, _ = compute_expected_free_energy(
            A=model.A,
            B=model.B,
            C=model.C,
            beliefs=model.beliefs,
            action=a
        )
    
    # Softmax for policy selection
    P = softmax(-temperature * G)
    return np.random.choice(len(P), p=P)
```

### In Visualization
```python
def plot_efe_components(model, action: int):
    """Visualize EFE components."""
    total, epist, prag = compute_expected_free_energy(
        A=model.A,
        B=model.B,
        C=model.C,
        beliefs=model.beliefs,
        action=action
    )
    
    # Create stacked bar plot
    plt.bar(['Total', 'Components'], 
            [total, 0],
            label='Total EFE')
    plt.bar(['Total', 'Components'],
            [0, epist],
            label='Epistemic')
    plt.bar(['Total', 'Components'],
            [0, prag],
            bottom=[0, epist],
            label='Pragmatic')
```

## Properties

### Mathematical Properties
- Non-negative epistemic value
- Pragmatic value depends on preferences
- Total EFE balances exploration/exploitation
- Links to [[free_energy_principle]]

### Computational Properties
- O(n²) complexity for n states
- Numerically stable with log preferences
- Parallelizable across actions
- Links to [[computational_complexity]]

## Visualization

### Key Plots
- [[efe_components]]: Epistemic vs Pragmatic
- [[efe_landscape]]: EFE surface over beliefs
- [[policy_evaluation]]: EFE for each action

## Related Implementations
- [[compute_vfe]]: Variational Free Energy
- [[update_beliefs]]: Belief updating
- [[select_policy]]: Policy selection

## References
- [[friston_2017]] - Active Inference
- [[da_costa_2020]] - Active Inference POMDP
- [[parr_2019]] - Generalizing Free Energy 