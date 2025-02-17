---
type: mathematical_concept
id: expected_free_energy_update_001
created: 2024-02-05
modified: 2024-02-05
tags: [mathematics, active-inference, policy-selection, free-energy]
aliases: [EFE-update, policy-prior-update]
---

# Expected Free Energy Update

## Mathematical Definition

The Expected Free Energy (G) for each action a is:

$G(a) = \underbrace{\mathbb{E}_{Q(o,s|a)}[\ln Q(s|a) - \ln P(s|a)]}_{\text{epistemic value}} + \underbrace{\mathbb{E}_{Q(o|a)}[-\ln P(o)]}_{\text{pragmatic value}}$

where:
- Q(s|a) is predicted state distribution under action a
- P(s|a) is prior state distribution
- Q(o|a) is predicted observation distribution
- P(o) is preferred observation distribution (from C matrix)

## Policy Prior Update

The policy prior E is updated using:

$E_{t+1} = (1-\alpha)E_t + \alpha\sigma(-\gamma G)$

where:
- $E_t$ is current policy prior
- $\alpha$ is learning rate (0 for static prior)
- $\gamma$ is precision parameter
- $\sigma$ is softmax function
- G is vector of Expected Free Energies

## Implementation

```python
def update_policy_prior(
    A: np.ndarray,           # Observation model P(o|s)
    B: np.ndarray,           # Transition model P(s'|s,a)
    C: np.ndarray,           # Log preferences ln P(o)
    E: np.ndarray,           # Current policy prior P(a)
    beliefs: np.ndarray,     # Current state beliefs Q(s)
    alpha: float = 0.1,      # Learning rate
    gamma: float = 1.0       # Precision
) -> np.ndarray:
    """Update policy prior using Expected Free Energy.
    
    Args:
        A: Observation likelihood matrix [n_obs x n_states]
        B: State transition tensor [n_states x n_states x n_actions]
        C: Log preference vector [n_obs]
        E: Current policy prior [n_actions]
        beliefs: Current belief state [n_states]
        alpha: Learning rate (0 for static prior)
        gamma: Precision parameter
        
    Returns:
        Updated policy prior E [n_actions]
    """
    n_actions = B.shape[2]
    G = np.zeros(n_actions)
    
    for a in range(n_actions):
        # Predicted next state distribution
        Qs_a = B[:, :, a] @ beliefs
        
        # Predicted observation distribution
        Qo_a = A @ Qs_a
        
        # Epistemic value (state uncertainty)
        epistemic = compute_entropy(Qs_a)
        
        # Pragmatic value (preference satisfaction)
        pragmatic = -np.sum(Qo_a * C)  # Negative because C is log preferences
        
        # Total Expected Free Energy
        G[a] = epistemic + pragmatic
    
    # Compute new policy distribution using softmax
    E_new = softmax(-gamma * G)
    
    # Update with learning rate
    E_updated = (1 - alpha) * E + alpha * E_new
    
    return E_updated
```

## Usage

The function is used in the Active Inference loop to update action priors:

```python
# Initialize uniform action prior
E = np.ones(n_actions) / n_actions

# In simulation loop:
E = update_policy_prior(
    A=model.A,
    B=model.B, 
    C=model.C,
    E=model.E,
    beliefs=model.state.beliefs,
    alpha=model.config['inference']['learning_rate'],
    gamma=model.config['inference']['temperature']
)
```

## Properties

### Mathematical Properties
- [[probability_conservation]] - Output is valid probability distribution
- [[policy_convergence]] - Converges to optimal policy under right conditions
- [[learning_dynamics]] - Controlled by learning rate and precision

### Computational Properties
- [[numerical_stability]] - Uses log space for preferences
- [[computational_efficiency]] - Vectorized operations
- [[memory_usage]] - O(n_actions) space complexity

## Related Concepts
- [[free_energy_principle]]
- [[active_inference]]
- [[policy_selection]]
- [[belief_updating]]

## References
- [[friston_2017]] - Mathematical foundations
- [[da_costa_2020]] - Active Inference implementation
- [[parr_2019]] - Policy learning 