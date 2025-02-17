---
type: mathematical_concept
id: information_gain_001
created: 2024-02-05
modified: 2024-02-05
tags: [information-theory, active-inference, uncertainty]
aliases: [information-gain, mutual-information, uncertainty-reduction]
---

# Information Gain

## Information Structure

```mermaid
graph TB
    IG[Information Gain] --> |Measures| KL[KL Divergence]
    IG --> |Quantifies| UR[Uncertainty Reduction]
    IG --> |Drives| EP[Epistemic Value]
    
    KL --> |Between| PB[Prior/Posterior]
    KL --> |Computes| DV[Divergence Value]
    
    UR --> |Through| OB[Observation]
    UR --> |Updates| BE[Beliefs]
    
    EP --> |Guides| AS[Action Selection]
    EP --> |Balances| EX[Exploration]
    
    classDef concept fill:#f9f,stroke:#333,stroke-width:2px
    classDef process fill:#bbf,stroke:#333,stroke-width:2px
    classDef outcome fill:#bfb,stroke:#333,stroke-width:2px
    
    class IG concept
    class KL,UR,EP process
    class PB,OB,AS,EX outcome
```

## Belief Update Flow

```mermaid
graph LR
    subgraph Prior Knowledge
        P[Prior P(s)] --> |Initial| H1[H(s)]
        H1 --> |Uncertainty| U1[Prior Uncertainty]
    end
    
    subgraph Observation
        O[Observation o] --> |Evidence| L[Likelihood P(o|s)]
        L --> |Bayes| Po[Posterior P(s|o)]
    end
    
    subgraph Information
        Po --> |Updated| H2[H(s|o)]
        H2 --> |Reduction| U2[Posterior Uncertainty]
        
        U1 --> |Difference| IG[Information Gain]
        U2 --> |Difference| IG
    end
    
    classDef state fill:#f9f,stroke:#333,stroke-width:2px
    classDef process fill:#bbf,stroke:#333,stroke-width:2px
    classDef measure fill:#bfb,stroke:#333,stroke-width:2px
    
    class P,O,Po state
    class L,H1,H2 process
    class U1,U2,IG measure
```

## Computation Dynamics

```mermaid
stateDiagram-v2
    [*] --> InitialState
    
    state "Information Processing" as IP {
        InitialState --> BeliefState
        BeliefState --> ObservationState
        ObservationState --> UpdatedState
        
        state BeliefState {
            PriorBelief --> EntropyComputation
            EntropyComputation --> InitialUncertainty
        }
        
        state ObservationState {
            Observation --> LikelihoodEvaluation
            LikelihoodEvaluation --> PosteriorComputation
        }
        
        state UpdatedState {
            PosteriorBelief --> GainComputation
            GainComputation --> InformationValue
        }
    }
    
    IP --> [*]: InformationProcessed
```

## Value Computation

```mermaid
graph TD
    subgraph Distributions
        P[Prior] --> |KL| D[Divergence]
        Q[Posterior] --> |KL| D
    end
    
    subgraph Components
        D --> |Expected| EV[Expected Value]
        D --> |Actual| AV[Actual Value]
    end
    
    subgraph Integration
        EV --> |Weight| W[Weighted Sum]
        AV --> |Weight| W
        W --> |Total| IG[Information Gain]
    end
    
    classDef dist fill:#f9f,stroke:#333,stroke-width:2px
    classDef comp fill:#bbf,stroke:#333,stroke-width:2px
    classDef value fill:#bfb,stroke:#333,stroke-width:2px
    
    class P,Q dist
    class D,EV,AV comp
    class W,IG value
```

## Analysis Methods

```mermaid
graph TD
    subgraph Input
        P[Prior] --> A[Analysis]
        O[Observation] --> A
        Q[Posterior] --> A
    end
    
    subgraph Computation
        A --> |Compute| IG[Information Gain]
        A --> |Measure| EN[Entropy Change]
        A --> |Evaluate| KL[KL Divergence]
    end
    
    subgraph Metrics
        IG --> |Track| EF[Effectiveness]
        EN --> |Monitor| PR[Progress]
        KL --> |Assess| CV[Convergence]
    end
    
    subgraph Output
        EF --> |Report| R[Results]
        PR --> |Visualize| V[Visualization]
        CV --> |Update| S[Strategy]
    end
    
    classDef input fill:#f9f,stroke:#333,stroke-width:2px
    classDef process fill:#bbf,stroke:#333,stroke-width:2px
    classDef output fill:#bfb,stroke:#333,stroke-width:2px
    
    class P,O,Q input
    class IG,EN,KL process
    class R,V,S output
```

## Mathematical Formulation

Information gain is defined as the KL divergence between posterior and prior distributions:

$IG(s;o) = D_{KL}[P(s|o)\|P(s)] = \mathbb{E}_{P(s|o)}[\ln P(s|o) - \ln P(s)]$

Links to:
- [[kl_divergence]] - Divergence measure
- [[bayesian_inference]] - Posterior computation
- [[entropy]] - Uncertainty measure

## Implementation

```python
def compute_information_gain(
    prior: np.ndarray,      # Prior distribution P(s)
    posterior: np.ndarray,  # Posterior distribution P(s|o)
    method: str = 'kl'     # Computation method
) -> float:
    """Compute information gain between distributions.
    
    Args:
        prior: Prior probability distribution
        posterior: Posterior probability distribution
        method: Method to use ('kl' or 'entropy')
        
    Returns:
        Information gain value
    """
    if method == 'kl':
        return np.sum(posterior * (np.log(posterior + 1e-10) - 
                                 np.log(prior + 1e-10)))
    elif method == 'entropy':
        prior_entropy = -np.sum(prior * np.log(prior + 1e-10))
        post_entropy = -np.sum(posterior * np.log(posterior + 1e-10))
        return prior_entropy - post_entropy
    else:
        raise ValueError(f"Unknown method: {method}")
```

Links to:
- [[numerical_methods]] - Implementation details
- [[probability_distributions]] - Distribution handling
- [[numerical_stability]] - Stability considerations

## Applications

### Active Inference
- Drives exploration in [[epistemic_value]]
- Guides [[action_selection]]
- Measures [[belief_updating]] effectiveness
- Links to:
  - [[exploration_exploitation]] - Balance
  - [[policy_selection]] - Action choice
  - [[efe_components]] - Value components

### Information Theory
- Quantifies [[mutual_information]]
- Measures [[entropy]] reduction
- Evaluates [[channel_capacity]]
- Links to:
  - [[information_theory_axioms]] - Foundations
  - [[information_geometry]] - Geometric view
  - [[information_flow]] - Dynamic aspects

## Properties

1. **Non-negativity**
   - Always ≥ 0 by [[jensen_inequality]]
   - = 0 iff distributions identical
   - Links to [[information_bounds]]

2. **Asymmetry**
   - Not symmetric in arguments
   - Order matters (posterior vs prior)
   - Links to [[divergence_measures]]

3. **Additivity**
   - Chain rule decomposition
   - Sequential information gains
   - Links to [[chain_rule_probability]]

## Related Concepts
- [[uncertainty_resolution]] - Resolution process
- [[active_learning]] - Learning strategy
- [[optimal_experiment_design]] - Design theory
- [[information_theory]] - Theoretical basis
- [[bayesian_inference]] - Statistical framework

## References
- [[cover_thomas_2006]] - Information Theory
- [[mackay_2003]] - Information Theory
- [[friston_2017]] - Active Inference
- [[lindley_1956]] - Information Measures 