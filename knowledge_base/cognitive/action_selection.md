---

title: Action Selection

type: concept

status: stable

created: 2024-02-06

updated: 2026-01-03

complexity: advanced

processing_priority: 1

tags:

  - cognition

  - decision_making

  - control

  - optimization

  - behavior

  - motor_control

  - planning

semantic_relations:

  - type: implements

    links:

      - [[active_inference]]

      - [[optimal_control]]

      - [[policy_selection]]

  - type: relates

    links:

      - [[decision_making]]

      - [[motor_control]]

      - [[planning]]

      - [[reinforcement_learning]]

  - type: mathematical_basis

    links:

      - [[expected_free_energy]]

      - [[path_integral_control]]

      - [[optimization_theory]]

---

## Overview

Action selection is a fundamental process in cognitive systems that involves choosing appropriate actions based on current beliefs, goals, and environmental context. In the active inference framework, action selection emerges from the principle of free energy minimization, where actions are selected to minimize expected free energy over future states.

> **Note**: [[policy_selection]] operates over a time horizon of 1 or more timesteps, while [[action_selection]] involves sampling a single action from a probability distribution ([[E_matrix]]), habit, or policy posterior.

## Mathematical Framework

### Expected Free Energy

The expected free energy $G(\pi)$ for a policy $\pi$ is defined as:

```math

G(\pi) = \sum_\tau G(\pi,\tau)

G(\pi,\tau) = E_{Q(o_\tau,s_\tau|\pi)}[\ln Q(s_\tau|\pi) - \ln P(o_\tau,s_\tau|\pi)]

```

Components:

- [[expected_free_energy_components]]

- [[policy_evaluation]]

- [[temporal_horizon]]

### Policy Selection

Actions are selected using a softmax function over expected free energy:

```math

P(\pi) = \sigma(-\gamma G(\pi))

\sigma(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

```

where:

- $\gamma$ is the precision parameter

- $\sigma$ is the softmax function

## Implementation Framework

### 1. Policy Evaluation

```python

class PolicyEvaluator:

    def __init__(self):

        # Components for policy evaluation

        self.components = {

            'state_estimation': StateEstimator(

                method='variational',

                horizon='adaptive'

            ),

            'outcome_prediction': OutcomePredictor(

                model='generative',

                uncertainty=True

            ),

            'value_computation': ValueComputer(

                metrics=['expected_free_energy', 'epistemic_value', 'pragmatic_value'],

                weights='adaptive'

            )

        }

    def evaluate_policy(self, policy, current_state):

        """Evaluate a policy starting from current state"""

        # Estimate future states

        future_states = self.components['state_estimation'].predict(

            current_state, policy)

        # Predict outcomes

        predicted_outcomes = self.components['outcome_prediction'].predict(

            future_states)

        # Compute value

        value = self.components['value_computation'].compute(

            future_states, predicted_outcomes)

        return value

```

### 2. Action Selection

```python

class ActionSelector:

    def __init__(self):

        # Selection components

        self.components = {

            'policy_prior': PolicyPrior(

                type='learned',

                adaptation='online'

            ),

            'precision_control': PrecisionControl(

                method='adaptive',

                bounds=['lower', 'upper']

            ),

            'selection_mechanism': SelectionMechanism(

                algorithm='softmax',

                temperature='dynamic'

            )

        }

    def select_action(self, policy_values):

        """Select action based on policy values"""

        # Apply prior

        prior_values = self.components['policy_prior'].apply(policy_values)

        # Control precision

        precision = self.components['precision_control'].compute(prior_values)

        # Select action

        action = self.components['selection_mechanism'].select(

            prior_values, precision)

        return action

```

### 3. Execution Control

```python

class ExecutionController:

    def __init__(self):

        # Execution components

        self.components = {

            'motor_control': MotorController(

                type='hierarchical',

                feedback=True

            ),

            'monitoring': ExecutionMonitor(

                metrics=['accuracy', 'efficiency'],

                adaptation=True

            ),

            'adaptation': ExecutionAdapter(

                learning='online',

                optimization='continuous'

            )

        }

    def execute_action(self, action):

        """Execute selected action"""

        # Generate motor commands

        commands = self.components['motor_control'].generate(action)

        # Monitor execution

        performance = self.components['monitoring'].track(commands)

        # Adapt execution

        self.components['adaptation'].update(performance)

        return performance

```

## Advanced Concepts

### 1. Hierarchical Selection

- [[hierarchical_policies]]

  - Temporal abstraction

  - Action composition

  - Goal decomposition

- [[option_frameworks]]

  - Skill learning

  - Transfer learning

  - Hierarchical control

### 2. Active Inference

- [[expected_free_energy]]

  - Epistemic value

  - Pragmatic value

  - Information gain

- [[belief_updating]]

  - State estimation

  - Parameter learning

  - Structure learning

### 3. Optimization Methods

- [[policy_optimization]]

  - Gradient methods

  - Evolution strategies

  - Reinforcement learning

- [[trajectory_optimization]]

  - Path integral control

  - Optimal control

  - Model predictive control

## Applications

### 1. Motor Control

- [[motor_planning]]

  - Movement generation

  - Sequence learning

  - Coordination

- [[sensorimotor_integration]]

  - Feedback control

  - Forward models

  - Inverse models

### 2. Decision Making

- [[value_based_choice]]

  - Reward processing

  - Risk assessment

  - Temporal discounting

- [[exploration_exploitation]]

  - Information seeking

  - Uncertainty reduction

  - Resource allocation

### 3. Cognitive Control

- [[executive_function]]

  - Task switching

  - Response inhibition

  - Working memory

- [[attention_control]]

  - Resource allocation

  - Priority setting

  - Focus maintenance

## Implementation Examples

### Basic Action Selection
```python
class BasicActionSelector:
    """Basic action selection using expected free energy minimization."""

    def __init__(self, n_actions, precision=1.0):
        self.n_actions = n_actions
        self.precision = precision
        self.action_history = []

    def select_action(self, expected_free_energies):
        """
        Select action using softmax over negative expected free energies.

        Parameters:
        - expected_free_energies: array of G values for each action

        Returns:
        - selected_action: chosen action index
        - action_probabilities: probability distribution over actions
        """
        # Apply precision weighting
        scaled_G = self.precision * expected_free_energies

        # Softmax to get action probabilities
        action_probabilities = self._softmax(-scaled_G)  # Negative because we minimize G

        # Sample action
        selected_action = np.random.choice(self.n_actions, p=action_probabilities)

        # Store for analysis
        self.action_history.append({
            'action': selected_action,
            'probabilities': action_probabilities,
            'expected_free_energies': expected_free_energies
        })

        return selected_action, action_probabilities

    def _softmax(self, x):
        """Numerically stable softmax."""
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
```

### Hierarchical Action Selection
```python
class HierarchicalActionSelector:
    """Hierarchical action selection with multiple time scales."""

    def __init__(self, hierarchy_levels=[3, 5, 8]):
        self.levels = hierarchy_levels
        self.temporal_horizons = [1, 4, 16]  # Increasing time horizons

        # Initialize level-specific selectors
        self.level_selectors = [
            BasicActionSelector(n_actions=level, precision=1.0)
            for level in hierarchy_levels
        ]

        # Cross-level communication
        self.level_weights = np.ones(len(hierarchy_levels)) / len(hierarchy_levels)

    def hierarchical_selection(self, observation, context=None):
        """
        Perform hierarchical action selection across multiple time scales.
        """
        level_actions = []
        level_probabilities = []

        # Process each level
        for level_idx, selector in enumerate(self.level_selectors):
            # Level-specific expected free energies
            level_G = self._compute_level_expected_free_energy(
                level_idx, observation, context
            )

            # Level-specific action selection
            action, probabilities = selector.select_action(level_G)

            level_actions.append(action)
            level_probabilities.append(probabilities)

        # Combine across levels
        combined_action = self._combine_level_actions(level_actions, level_probabilities)

        return combined_action, {
            'level_actions': level_actions,
            'level_probabilities': level_probabilities,
            'combined_action': combined_action
        }

    def _compute_level_expected_free_energy(self, level_idx, observation, context):
        """Compute expected free energy for a specific hierarchical level."""
        horizon = self.temporal_horizons[level_idx]
        n_actions = self.levels[level_idx]

        G_values = np.zeros(n_actions)

        for action in range(n_actions):
            # Simulate policy over horizon
            policy = [action] * horizon  # Simplified: repeat action

            # Compute expected free energy for this policy
            G = self._evaluate_policy(policy, observation, context, level_idx)
            G_values[action] = G

        return G_values

    def _evaluate_policy(self, policy, observation, context, level_idx):
        """Evaluate a policy's expected free energy."""
        # Simplified policy evaluation
        # In practice, this would involve forward simulation
        G = 0
        for t, action in enumerate(policy):
            # Epistemic value (information gain)
            epistemic = self._compute_epistemic_value(action, level_idx)

            # Pragmatic value (preference satisfaction)
            pragmatic = self._compute_pragmatic_value(action, observation, context, level_idx)

            # Temporal discounting
            discount = 0.9 ** t

            G += discount * (epistemic + pragmatic)

        return G

    def _compute_epistemic_value(self, action, level_idx):
        """Compute information gain for action at given level."""
        # Simplified: higher-level actions have more epistemic value
        return 0.1 * (level_idx + 1) * np.random.randn()

    def _compute_pragmatic_value(self, action, observation, context, level_idx):
        """Compute preference satisfaction for action."""
        # Simplified: context-dependent preferences
        if context is not None:
            preference_match = np.dot(observation, context.get('preferences', observation))
        else:
            preference_match = np.dot(observation, observation)  # Self-preference

        return -preference_match  # Negative because we minimize G

    def _combine_level_actions(self, level_actions, level_probabilities):
        """Combine actions across hierarchical levels."""
        # Weighted combination based on level weights
        combined_probabilities = np.zeros_like(level_probabilities[0])

        for level_idx, (action, probs) in enumerate(zip(level_actions, level_probabilities)):
            weight = self.level_weights[level_idx]
            combined_probabilities += weight * probs

        # Normalize
        combined_probabilities = combined_probabilities / np.sum(combined_probabilities)

        # Sample combined action
        combined_action = np.random.choice(len(combined_probabilities), p=combined_probabilities)

        return combined_action
```

### Advanced Selection Mechanisms

#### Risk-Sensitive Action Selection
```python
class RiskSensitiveActionSelector(BasicActionSelector):
    """Action selection with risk sensitivity."""

    def __init__(self, n_actions, precision=1.0, risk_preference=0.0):
        super().__init__(n_actions, precision)
        self.risk_preference = risk_preference  # 0: risk-neutral, >0: risk-seeking, <0: risk-averse

    def select_action_risk_sensitive(self, expected_free_energies, variances):
        """
        Risk-sensitive action selection using mean-variance optimization.
        """
        # Compute risk-adjusted values
        risk_adjusted_G = expected_free_energies + self.risk_preference * variances

        # Apply precision weighting
        scaled_G = self.precision * risk_adjusted_G

        # Softmax selection
        action_probabilities = self._softmax(-scaled_G)

        selected_action = np.random.choice(self.n_actions, p=action_probabilities)

        return selected_action, action_probabilities
```

#### Multi-Objective Action Selection
```python
class MultiObjectiveActionSelector:
    """Action selection balancing multiple objectives."""

    def __init__(self, n_actions, objectives=['epistemic', 'pragmatic', 'efficiency']):
        self.n_actions = n_actions
        self.objectives = objectives
        self.weights = np.ones(len(objectives)) / len(objectives)  # Equal weighting initially

    def select_action_multiobjective(self, objective_values):
        """
        Multi-objective action selection.

        Parameters:
        - objective_values: dict with objective names as keys and arrays as values
        """
        # Compute weighted combination
        combined_values = np.zeros(self.n_actions)

        for obj_name, values in objective_values.items():
            if obj_name in self.objectives:
                obj_idx = self.objectives.index(obj_name)
                weight = self.weights[obj_idx]
                combined_values += weight * values

        # Standard softmax selection
        action_probabilities = self._softmax(-combined_values)
        selected_action = np.random.choice(self.n_actions, p=action_probabilities)

        return selected_action, action_probabilities

    def adapt_weights(self, feedback):
        """Adapt objective weights based on performance feedback."""
        # Simplified weight adaptation
        for i, obj in enumerate(self.objectives):
            if obj in feedback:
                # Increase weight for objectives that led to good outcomes
                self.weights[i] *= (1 + 0.1 * feedback[obj])

        # Renormalize
        self.weights = self.weights / np.sum(self.weights)

    def _softmax(self, x):
        """Numerically stable softmax."""
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
```

## Research Directions

### 1. Theoretical Extensions

- [[quantum_decision_making]]

  - Quantum probability

  - Interference effects

  - Entanglement

- [[stochastic_control]]

  - Risk sensitivity

  - Noise adaptation

  - Robustness

### 2. Applications

- [[robotics]]

  - Manipulation

  - Navigation

  - Human-robot interaction

- [[clinical_applications]]

  - Movement disorders

  - Decision pathologies

  - Rehabilitation

### 3. Methods Development

- [[deep_active_inference]]

  - Neural architectures

  - Learning algorithms

  - Scaling solutions

- [[adaptive_control]]

  - Online learning

  - Meta-learning

  - Transfer learning

## References

- [[friston_2017]] - "Active Inference and Learning"

- [[parr_friston_2019]] - "Generalised Free Energy and Active Inference"

- [[da_costa_2020]] - "Active inference, stochastic control, and expected free energy"

- [[tschantz_2020]] - "Scaling active inference"

## See Also

- [[active_inference]]

- [[optimal_control]]

- [[reinforcement_learning]]

- [[motor_control]]

- [[decision_making]]

- [[planning]]

- [[cognitive_control]]

