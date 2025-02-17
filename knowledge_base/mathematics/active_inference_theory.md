# Active Inference Theory

---
type: mathematical_concept
id: active_inference_theory_001
created: 2024-02-06
modified: 2024-02-06
tags: [mathematics, active-inference, free-energy, cognitive-modeling]
aliases: [active-inference-framework, free-energy-principle]
semantic_relations:
  - type: implements
    links: 
      - [[../../docs/research/research_documentation_index|Research Documentation]]
      - [[free_energy_theory]]
  - type: uses
    links:
      - [[variational_methods]]
      - [[information_theory]]
      - [[optimal_control]]
  - type: documented_by
    links:
      - [[../../docs/guides/implementation_guides_index|Implementation Guides]]
      - [[../../docs/api/api_documentation_index|API Documentation]]
---

## Overview

Active Inference unifies perception, learning, and action under the free energy principle. This document provides a comprehensive mathematical treatment of Active Inference and its applications in cognitive modeling.

## Core Mathematics

### Generative Model
```python
class GenerativeModel:
    """
    Generative model implementation.
    
    Theory:
        - [[state_space_model]]
        - [[markov_blanket]]
        - [[hierarchical_model]]
    Mathematics:
        - [[probability_theory]]
        - [[graphical_models]]
    """
    def __init__(self,
                 transition_model: TransitionDist,
                 observation_model: ObservationDist,
                 prior_beliefs: PriorDist):
        self.P_transition = transition_model
        self.P_observation = observation_model
        self.P_prior = prior_beliefs
    
    def joint_probability(self,
                        states: np.ndarray,
                        observations: np.ndarray) -> float:
        """Compute joint probability P(o,s)."""
        return (self.P_observation.prob(observations, states) *
                self.P_transition.prob(states) *
                self.P_prior.prob(states[0]))
    
    def generate_sequence(self,
                        policy: Policy,
                        horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate state-observation sequence."""
        states = self._generate_states(policy, horizon)
        observations = self._generate_observations(states)
        return states, observations
```

### Variational Inference
```python
class VariationalInference:
    """
    Variational inference for state estimation.
    
    Theory:
        - [[variational_bayes]]
        - [[mean_field]]
        - [[message_passing]]
    Mathematics:
        - [[variational_methods]]
        - [[information_geometry]]
    """
    def __init__(self,
                 generative_model: GenerativeModel,
                 recognition_model: RecognitionModel):
        self.gen_model = generative_model
        self.rec_model = recognition_model
        
    def update_beliefs(self,
                      observation: np.ndarray,
                      initial_belief: np.ndarray) -> np.ndarray:
        """Update beliefs using variational inference."""
        current_belief = initial_belief.copy()
        
        while not self._converged():
            # Compute expected energy
            energy = self._expected_energy(
                observation, current_belief
            )
            
            # Compute entropy
            entropy = self._entropy(current_belief)
            
            # Update beliefs
            current_belief = self._normalize(
                np.exp(-(energy - entropy))
            )
        
        return current_belief
```

### Expected Free Energy
```python
class ExpectedFreeEnergy:
    """
    Expected free energy computation.
    
    Theory:
        - [[expected_free_energy]]
        - [[epistemic_value]]
        - [[pragmatic_value]]
    Mathematics:
        - [[information_theory]]
        - [[decision_theory]]
    """
    def __init__(self,
                 generative_model: GenerativeModel,
                 preferences: Preferences):
        self.model = generative_model
        self.C = preferences
    
    def compute_efe(self,
                   belief_state: np.ndarray,
                   policy: Policy) -> float:
        """
        Compute expected free energy for policy.
        G = E_Q[ln Q(s') - ln P(o',s')]
        """
        # Epistemic value (information gain)
        epistemic = self._compute_information_gain(
            belief_state, policy
        )
        
        # Pragmatic value (expected utility)
        pragmatic = self._compute_expected_utility(
            belief_state, policy
        )
        
        return epistemic + pragmatic
    
    def _compute_information_gain(self,
                                belief_state: np.ndarray,
                                policy: Policy) -> float:
        """
        Compute expected information gain.
        IG = E_Q[ln Q(s'|o') - ln Q(s')]
        """
        # Predictive posterior
        Q_posterior = self._compute_predictive_posterior(
            belief_state, policy
        )
        
        # Prior predictive
        Q_prior = self._compute_prior_predictive(
            belief_state, policy
        )
        
        return self._expected_kl(Q_posterior, Q_prior)
    
    def _compute_expected_utility(self,
                                belief_state: np.ndarray,
                                policy: Policy) -> float:
        """
        Compute expected utility.
        U = E_Q[ln P(o')]
        """
        # Expected observations
        expected_obs = self._compute_expected_observations(
            belief_state, policy
        )
        
        # Compute utility using preferences
        return np.sum(self.C.utility(expected_obs))
```

## Policy Selection

### Policy Space
```python
class PolicySpace:
    """
    Policy space implementation.
    
    Theory:
        - [[policy_selection]]
        - [[action_space]]
        - [[planning]]
    Mathematics:
        - [[optimal_control]]
        - [[decision_theory]]
    """
    def __init__(self,
                 action_space: ActionSpace,
                 horizon: int):
        self.actions = action_space
        self.horizon = horizon
        self.policies = self._construct_policies()
    
    def evaluate_policies(self,
                        belief_state: np.ndarray,
                        efe: ExpectedFreeEnergy) -> np.ndarray:
        """Evaluate all policies using EFE."""
        return np.array([
            efe.compute_efe(belief_state, pi)
            for pi in self.policies
        ])
    
    def select_policy(self,
                     policy_values: np.ndarray,
                     temperature: float = 1.0) -> Policy:
        """Select policy using softmax."""
        p = softmax(-temperature * policy_values)
        return self.policies[np.argmax(p)]
```

### Action Selection
```python
class ActionSelection:
    """
    Action selection implementation.
    
    Theory:
        - [[active_inference]]
        - [[policy_selection]]
        - [[exploration_exploitation]]
    Mathematics:
        - [[decision_theory]]
        - [[information_theory]]
    """
    def __init__(self,
                 policy_space: PolicySpace,
                 efe: ExpectedFreeEnergy):
        self.policies = policy_space
        self.efe = efe
    
    def select_action(self,
                     belief_state: np.ndarray,
                     temperature: float = 1.0) -> Action:
        """Select action using active inference."""
        # Evaluate policies
        G = self.policies.evaluate_policies(
            belief_state, self.efe
        )
        
        # Select policy
        policy = self.policies.select_policy(G, temperature)
        
        # Return first action
        return policy.first_action()
```

## Learning

### Parameter Learning
```python
class ParameterLearning:
    """
    Parameter learning implementation.
    
    Theory:
        - [[variational_learning]]
        - [[empirical_bayes]]
        - [[hierarchical_inference]]
    Mathematics:
        - [[variational_methods]]
        - [[information_geometry]]
    """
    def __init__(self,
                 model: GenerativeModel,
                 prior: ParameterPrior):
        self.model = model
        self.prior = prior
    
    def update_parameters(self,
                        observations: np.ndarray,
                        states: np.ndarray) -> None:
        """Update model parameters."""
        # Compute sufficient statistics
        stats = self._compute_sufficient_statistics(
            observations, states
        )
        
        # Update parameters using conjugate updates
        self._conjugate_parameter_update(stats)
```

## Information Geometry

### Natural Gradients
```python
class NaturalGradient:
    """
    Natural gradient implementation.
    
    Theory:
        - [[information_geometry]]
        - [[fisher_information]]
        - [[natural_gradient]]
    Mathematics:
        - [[riemannian_geometry]]
        - [[differential_geometry]]
    """
    def __init__(self,
                 manifold: StatisticalManifold,
                 metric: FisherMetric):
        self.manifold = manifold
        self.metric = metric
    
    def compute_natural_gradient(self,
                               params: np.ndarray,
                               grad: np.ndarray) -> np.ndarray:
        """Compute natural gradient."""
        # Compute Fisher information
        G = self.metric.fisher_information(params)
        
        # Compute natural gradient
        return np.linalg.solve(G, grad)
```

## Mathematical Properties

### Free Energy Decomposition
```python
# @free_energy_decomposition
decomposition = {
    "variational": {
        "energy": "E_Q[ln P(o,s)]",
        "entropy": "E_Q[ln Q(s)]",
        "surprise": "ln P(o)"
    },
    "expected": {
        "epistemic": "E_Q[ln Q(s'|o') - ln Q(s')]",
        "pragmatic": "E_Q[ln P(o')]",
        "complexity": "KL[Q(s')||P(s')]"
    }
}
```

### Theoretical Guarantees
```python
# @theoretical_guarantees
guarantees = {
    "perception": {
        "bound": "F ≥ -ln P(o)",
        "optimality": "Q*(s) = P(s|o)",
        "convergence": "Local minimum of F"
    },
    "learning": {
        "bound": "F ≥ -ln P(o|θ)",
        "optimality": "θ* = argmax P(o|θ)",
        "convergence": "EM-like convergence"
    },
    "action": {
        "bound": "G ≥ E_Q[ln P(o')]",
        "optimality": "π* = argmin G",
        "exploration": "Automatic exploration-exploitation"
    }
}
```

## Implementation Considerations

### Numerical Methods
```python
# @numerical_methods
numerical_implementations = {
    "inference": {
        "gradient_descent": "Natural gradient VI",
        "message_passing": "Belief propagation",
        "sampling": "Importance sampling"
    },
    "learning": {
        "conjugate_updates": "Empirical Bayes",
        "gradient_ascent": "Parameter optimization",
        "em_algorithm": "Expectation-maximization"
    },
    "action": {
        "policy_search": "Direct policy search",
        "planning": "Tree search methods",
        "online": "Real-time updates"
    }
}
```

## Documentation Links
- [[../../docs/research/research_documentation_index|Research Documentation]]
- [[../../docs/guides/implementation_guides_index|Implementation Guides]]
- [[../../docs/api/api_documentation_index|API Documentation]]
- [[../../docs/examples/usage_examples_index|Usage Examples]]

## References
- [[friston]] - Active Inference
- [[parr]] - Mathematical Foundations
- [[da_costa]] - Information Geometry
- [[bogacz]] - Tutorial Review 