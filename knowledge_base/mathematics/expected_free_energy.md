---
type: mathematical_concept
id: expected_free_energy_001
created: 2024-02-05
modified: 2024-03-15
tags: [mathematics, active-inference, free-energy, policy-selection, decision-theory]
aliases: [EFE, expected-free-energy, policy-selection-objective]
complexity: advanced
processing_priority: 1
semantic_relations:
  - type: implements
    links: 
      - [[active_inference]]
      - [[policy_selection]]
  - type: mathematical_basis
    links:
      - [[information_theory]]
      - [[decision_theory]]
      - [[optimization_theory]]
  - type: relates
    links:
      - [[free_energy]]
      - [[path_integral_free_energy]]
      - [[active_inference_control]]
---

# Expected Free Energy

## Mathematical Framework

### Core Definition
The expected free energy $G(\pi)$ for a policy $\pi$ is defined as:

$G(\pi) = \mathbb{E}_{Q(o',s'|\pi)}[\ln Q(s'|\pi) - \ln P(o',s'|\pi)]$

where:
- $Q(s'|\pi)$ is the predicted state distribution under policy $\pi$
- $P(o',s'|\pi)$ is the generative model of future outcomes
- $\mathbb{E}_{Q}$ denotes expectation under $Q$

### Decomposition
The expected free energy can be decomposed into:

$G(\pi) = \underbrace{\mathbb{E}_{Q}[\ln Q(s'|\pi) - \ln P(s'|o',\pi)]}_{\text{Information Gain}} - \underbrace{\mathbb{E}_{Q}[\ln P(o'|\pi)]}_{\text{Expected Value}}$

### Policy Selection
The optimal policy distribution is given by:

$P(\pi) = \sigma(-\gamma G(\pi))$

where:
- $\sigma$ is the softmax function
- $\gamma$ is the precision parameter

## Components

### 1. Information Gain
- Epistemic value
- Uncertainty reduction
- Exploration drive
- Precision weighting

### 2. Expected Value
- Pragmatic value
- Goal achievement
- Preference satisfaction
- Utility maximization

### 3. Policy Precision
- Decision temperature
- Exploration-exploitation
- Confidence scaling
- Adaptive control

## Advanced Implementation

### 1. Policy Evaluation
```python
class PolicyEvaluator:
    def __init__(self):
        self.components = {
            'information': InformationGainComputer(
                method='mutual_information',
                approximation='monte_carlo'
            ),
            'value': ExpectedValueComputer(
                method='path_integral',
                horizon='adaptive'
            ),
            'precision': PrecisionOptimizer(
                method='gradient',
                adaptation='online'
            )
        }
    
    def evaluate_policy(
        self,
        policy: Policy,
        model: GenerativeModel,
        horizon: int
    ) -> Tuple[float, dict]:
        """Evaluate policy using expected free energy"""
        # Compute information gain
        info_gain = self.components['information'].compute(
            policy, model, horizon)
            
        # Compute expected value
        exp_value = self.components['value'].compute(
            policy, model, horizon)
            
        # Optimize precision
        precision = self.components['precision'].optimize(
            info_gain, exp_value)
            
        # Combine terms
        G = precision * (info_gain - exp_value)
        
        metrics = {
            'information_gain': info_gain,
            'expected_value': exp_value,
            'precision': precision
        }
        
        return G, metrics
```

### 2. Path Integration
```python
class PathIntegrator:
    def __init__(self):
        self.components = {
            'dynamics': StateTransitionModel(
                type='stochastic',
                integration='euler'
            ),
            'observation': ObservationModel(
                type='probabilistic',
                noise='adaptive'
            ),
            'accumulator': PathAccumulator(
                method='importance_sampling',
                particles='adaptive'
            )
        }
    
    def integrate_path(
        self,
        policy: Policy,
        model: GenerativeModel,
        horizon: int
    ) -> Tuple[float, dict]:
        """Compute path integral of expected free energy"""
        # Initialize path
        path = self.components['dynamics'].initialize(policy)
        
        # Integrate over horizon
        for t in range(horizon):
            # Propagate state
            state = self.components['dynamics'].step(
                path, policy, t)
                
            # Generate observation
            obs = self.components['observation'].generate(
                state, policy)
                
            # Accumulate free energy
            path = self.components['accumulator'].update(
                path, state, obs)
        
        return path.total_energy, path.metrics
```

### 3. Policy Distribution
```python
class PolicyDistribution:
    def __init__(self):
        self.components = {
            'evaluator': PolicyEvaluator(
                method='expected_free_energy',
                horizon='adaptive'
            ),
            'selector': PolicySelector(
                method='softmax',
                temperature='adaptive'
            ),
            'optimizer': DistributionOptimizer(
                method='natural_gradient',
                constraints='probability'
            )
        }
    
    def compute_distribution(
        self,
        policies: List[Policy],
        model: GenerativeModel
    ) -> Tuple[Distribution, dict]:
        """Compute policy distribution"""
        # Evaluate policies
        evaluations = [
            self.components['evaluator'].evaluate(pi, model)
            for pi in policies
        ]
        
        # Select policies
        selection = self.components['selector'].select(
            evaluations)
            
        # Optimize distribution
        distribution = self.components['optimizer'].optimize(
            selection)
            
        return distribution
```

## Advanced Concepts

### 1. Information Theory
- [[mutual_information]]
  - State-observation coupling
  - Uncertainty reduction
- [[kl_divergence]]
  - Policy divergence
  - Distribution matching

### 2. Decision Theory
- [[utility_theory]]
  - Preference encoding
  - Value functions
- [[risk_sensitivity]]
  - Risk aversion
  - Uncertainty handling

### 3. Control Theory
- [[optimal_control]]
  - Trajectory optimization
  - Cost minimization
- [[stochastic_control]]
  - Noise handling
  - Robust control

## Applications

### 1. Planning
- [[hierarchical_planning]]
  - Task decomposition
  - Abstract reasoning
- [[model_predictive_control]]
  - Receding horizon
  - Online optimization

### 2. Learning
- [[exploration_exploitation]]
  - Strategy adaptation
  - Knowledge acquisition
- [[meta_learning]]
  - Policy adaptation
  - Transfer learning

### 3. Decision Making
- [[active_sensing]]
  - Information seeking
  - Attention allocation
- [[goal_directed_behavior]]
  - Preference satisfaction
  - Task completion

## Research Directions

### 1. Theoretical Extensions
- [[quantum_decision_theory]]
  - Quantum probabilities
  - Interference effects
- [[relativistic_decision_theory]]
  - Causal decision theory
  - Temporal consistency

### 2. Computational Methods
- [[deep_active_inference]]
  - Neural architectures
  - End-to-end learning
- [[symbolic_planning]]
  - Logical reasoning
  - Program synthesis

### 3. Applications
- [[robotics_planning]]
  - Motion planning
  - Task execution
- [[cognitive_architectures]]
  - Decision making
  - Behavior generation

## References
- [[friston_2015]] - "Active inference and epistemic value"
- [[parr_2019]] - "Generalised free energy and active inference"
- [[da_costa_2020]] - "Active inference on discrete state-spaces"
- [[tschantz_2020]] - "Scaling active inference"

## See Also
- [[active_inference]]
- [[free_energy]]
- [[policy_selection]]
- [[decision_theory]]
- [[optimal_control]]
- [[planning_theory]]
- [[learning_theory]] 