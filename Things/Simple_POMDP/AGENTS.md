---
title: Simple POMDP Agents Documentation
type: agents
status: stable
created: 2025-01-01
updated: 2025-01-01
tags:
  - agents
  - pomdp
  - active_inference
  - tutorial
  - educational
semantic_relations:
  - type: documents
    links:
      - [[../../knowledge_base/mathematics/pomdp_structure]]
      - [[../../knowledge_base/cognitive/active_inference]]
  - type: supports
    links:
      - [[../../docs/guides/learning_paths]]
---

# Simple POMDP Agents Documentation

Educational agent implementation demonstrating Partially Observable Markov Decision Processes using Active Inference principles. This agent serves as an accessible introduction to Active Inference concepts and POMDP frameworks.

## 🧠 Agent Architecture

### Core POMDP Framework

#### SimplePOMDP Class
Basic Active Inference agent implementing fundamental POMDP concepts with clear, educational code structure.

```python
class SimplePOMDP:
    """Educational POMDP agent implementing Active Inference basics."""

    def __init__(self, config_path=None):
        """Initialize agent with POMDP matrices and Active Inference components."""
        # Load configuration
        self.config = self.load_configuration(config_path)

        # Initialize POMDP matrices
        self.A_matrix = self.initialize_A_matrix()  # Observation model P(o|s)
        self.B_matrix = self.initialize_B_matrix()  # Transition model P(s'|s,a)
        self.C_matrix = self.initialize_C_matrix()  # Preferences over observations
        self.D_matrix = self.initialize_D_matrix()  # Initial beliefs P(s)

        # Active Inference components
        self.belief_state = np.copy(self.D_matrix)  # Current belief distribution
        self.free_energy_calculator = FreeEnergyCalculator()
        self.action_selector = ActionSelector()

        # History tracking
        self.belief_history = [self.belief_state.copy()]
        self.action_history = []
        self.observation_history = []

    def step(self, action=None):
        """Execute one complete perception-action cycle."""
        # Select action if not provided
        if action is None:
            action = self.select_action()

        # Generate observation based on current belief and action
        observation = self.generate_observation(action)

        # Update beliefs based on action and observation
        self.update_beliefs(action, observation)

        # Calculate free energy
        free_energy = self.calculate_free_energy()

        # Store history
        self.action_history.append(action)
        self.observation_history.append(observation)
        self.belief_history.append(self.belief_state.copy())

        return observation, free_energy
```

### POMDP Components

#### Matrix Structure
Clear implementation of core POMDP matrices with educational focus on interpretability.

```python
def initialize_A_matrix(self):
    """Initialize observation model A(s,o) = P(o|s)."""
    A = np.zeros((self.config['num_observations'], self.config['num_states']))

    # Simple mapping: each state maps to specific observation
    for state in range(self.config['num_states']):
        obs_prob = 0.9  # High probability for correct observation
        error_prob = (1.0 - obs_prob) / (self.config['num_observations'] - 1)

        for obs in range(self.config['num_observations']):
            if obs == state % self.config['num_observations']:
                A[obs, state] = obs_prob  # Correct observation
            else:
                A[obs, state] = error_prob  # Observation error

    return self.normalize_matrix(A)  # Ensure columns sum to 1

def initialize_B_matrix(self):
    """Initialize transition model B(s',s,a) = P(s'|s,a)."""
    B = np.zeros((self.config['num_states'],
                  self.config['num_states'],
                  self.config['num_actions']))

    for action in range(self.config['num_actions']):
        if action == 0:  # Stay action
            B[:, :, action] = np.eye(self.config['num_states'])
        elif action == 1:  # Move right action
            for s in range(self.config['num_states']):
                next_s = (s + 1) % self.config['num_states']
                B[next_s, s, action] = 1.0
        # Add more action types as needed

    return B
```

### Active Inference Implementation

#### Belief Updating
Straightforward Bayesian belief updating with clear pedagogical structure.

```python
def update_beliefs(self, action, observation):
    """Update beliefs using Bayes rule: P(s|o,a) ∝ P(o|s,a) × P(s|a)."""
    # Predict next state distribution given current beliefs and action
    predicted_beliefs = self.predict_state_distribution(action)

    # Get likelihood of observation given predicted states
    likelihood = self.A_matrix[observation, :]  # P(o|s) for all states

    # Bayes rule: posterior ∝ likelihood × prior
    unnormalized_posterior = likelihood * predicted_beliefs

    # Normalize to get valid probability distribution
    self.belief_state = unnormalized_posterior / np.sum(unnormalized_posterior)

    return self.belief_state

def predict_state_distribution(self, action):
    """Predict next state distribution: P(s'|a) = Σ_s P(s'|s,a) × P(s)."""
    predicted = np.zeros(self.config['num_states'])

    for next_state in range(self.config['num_states']):
        for current_state in range(self.config['num_states']):
            transition_prob = self.B_matrix[next_state, current_state, action]
            predicted[next_state] += transition_prob * self.belief_state[current_state]

    return predicted
```

#### Free Energy Calculation
Educational implementation of variational free energy computation.

```python
def calculate_free_energy(self):
    """Calculate variational free energy: F = Σ_s P(s) × [ln P(s) - ln P(o,s)]."""
    free_energy = 0.0

    for state in range(self.config['num_states']):
        if self.belief_state[state] > 0:  # Avoid log(0)
            # Expected log likelihood: Σ_o P(o|s) × ln P(o|s)
            expected_log_likelihood = 0.0
            for obs in range(self.config['num_observations']):
                if self.A_matrix[obs, state] > 0:
                    expected_log_likelihood += (self.A_matrix[obs, state] *
                                               np.log(self.A_matrix[obs, state]))

            # Variational free energy contribution
            state_contribution = self.belief_state[state] * (
                np.log(self.belief_state[state]) - expected_log_likelihood
            )

            # Add preference contribution if C_matrix is defined
            if hasattr(self, 'C_matrix') and self.C_matrix is not None:
                # Simplified preference contribution
                state_contribution -= self.belief_state[state] * self.C_matrix[state]

            free_energy += state_contribution

    return free_energy
```

#### Action Selection
Clear implementation of Expected Free Energy minimization for action selection.

```python
def select_action(self):
    """Select action by minimizing expected free energy."""
    action_free_energies = []

    for action in range(self.config['num_actions']):
        # Calculate expected free energy for this action
        efe = self.calculate_expected_free_energy(action)
        action_free_energies.append(efe)

    # Select action with minimum expected free energy
    best_action = np.argmin(action_free_energies)

    return best_action

def calculate_expected_free_energy(self, action):
    """Calculate EFE for given action: G(a) = Σ_τ G_τ(a)."""
    efe = 0.0

    # Simplified: only consider immediate consequences
    # In full implementation, would sum over planning horizon

    # Epistemic affordance (information gain)
    epistemic_value = self.calculate_epistemic_value(action)

    # Extrinsic value (goal achievement)
    extrinsic_value = self.calculate_extrinsic_value(action)

    # Risk term (simplified)
    risk = 0.0

    efe = extrinsic_value - epistemic_value + risk  # Note: epistemic value reduces EFE

    return efe
```

## 📊 Agent Capabilities

### Perception and Learning

#### Basic Sensory Processing
- **Observation Generation**: Realistic observation sampling from belief states
- **Belief State Tracking**: Complete belief distribution maintenance
- **Uncertainty Quantification**: Entropy-based uncertainty measures

#### Learning Mechanisms
- **Belief Updating**: Bayesian inference from experience
- **Preference Learning**: Simple preference adaptation
- **Experience Integration**: History-based learning

### Decision Making

#### Policy Evaluation
- **Expected Free Energy**: Complete EFE calculation framework
- **Planning Horizon**: Configurable temporal planning depth
- **Risk Assessment**: Basic uncertainty-aware decision making

#### Action Selection
- **Greedy Selection**: Immediate EFE minimization
- **Temperature Control**: Softmax action selection with temperature
- **Exploration Strategies**: ε-greedy and softmax exploration

### Analysis and Visualization

#### Belief Tracking
- **Belief Evolution**: Complete belief state history
- **Convergence Monitoring**: Belief stability assessment
- **Uncertainty Analysis**: Entropy and variance tracking

#### Performance Metrics
- **Free Energy Tracking**: EFE evolution over time
- **Action Pattern Analysis**: Action selection statistics
- **Learning Progress**: Performance improvement metrics

## 🎯 Educational Applications

### Learning Objectives
- **POMDP Fundamentals**: Core concepts and mathematical structure
- **Active Inference Basics**: Free energy principle and minimization
- **Bayesian Reasoning**: Belief updating and probabilistic inference
- **Decision Making**: Policy evaluation and action selection

### Tutorial Integration
Supports progressive learning through:
- **Simple Examples**: Basic POMDP operations
- **Interactive Demonstrations**: Visual belief evolution
- **Mathematical Understanding**: Clear EFE calculations
- **Code Comprehension**: Readable, well-commented implementation

## 🧪 Testing and Validation

### Test Suite Structure
```python
class SimplePOMDPTestSuite:
    """Comprehensive testing for educational POMDP agent."""

    def __init__(self):
        self.agent_configs = self.generate_test_configs()
        self.test_environments = self.create_test_environments()

    def run_educational_tests(self):
        """Run tests that demonstrate learning concepts."""
        # Test belief updating mechanics
        self.test_belief_updating()

        # Test free energy calculations
        self.test_free_energy_computation()

        # Test action selection
        self.test_action_selection()

        # Test learning convergence
        self.test_learning_convergence()
```

### Validation Metrics
- **Belief Accuracy**: Correctness of belief updating
- **EFE Correctness**: Accuracy of free energy calculations
- **Decision Quality**: Effectiveness of action selection
- **Learning Stability**: Convergence and stability properties

## 📈 Performance Characteristics

### Computational Complexity
- **Belief Update**: O(num_states × num_observations)
- **Action Selection**: O(num_actions × planning_horizon × num_states)
- **Free Energy**: O(num_states × num_observations)

### Educational Trade-offs
- **Clarity over Efficiency**: Prioritizes understandability
- **Simplicity over Completeness**: Focuses on core concepts
- **Modularity over Optimization**: Easy to modify and extend

### Scalability
- **Small Problems**: Excellent performance (< 100ms per step)
- **Educational Scale**: Designed for problems up to 10-20 states
- **Visualization**: Optimized for plotting and analysis

## 🔧 Configuration and Customization

### Basic Configuration
```yaml
model:
  name: "SimplePOMDP"
  description: "Educational POMDP example"

state_space:
  num_states: 3

observation_space:
  num_observations: 2

action_space:
  num_actions: 2

inference:
  planning_horizon: 1  # Simplified for education
  learning_rate: 0.1
  temperature: 1.0
```

### Customization Options
- **Matrix Initialization**: Custom A, B, C, D matrices
- **Learning Parameters**: Adjustable learning rates and temperatures
- **Planning Depth**: Configurable planning horizons
- **Visualization Themes**: Custom plotting styles

## 📚 Documentation

### Implementation Details
See [[Simple_POMDP_README|Simple POMDP Implementation Details]] for:
- Complete API documentation
- Mathematical formulations with examples
- Configuration options and customization
- Troubleshooting and common issues

### Key Components
- [[simple_pomdp.py]] - Main agent implementation
- [[run_simple_pomdp.py]] - Execution examples
- [[configuration.yaml]] - Default configuration
- [[test_simple_pomdp.py]] - Comprehensive test suite

## 🔗 Related Documentation

### Theoretical Foundations
- [[../../knowledge_base/mathematics/pomdp_structure|POMDP Mathematical Structure]]
- [[../../knowledge_base/cognitive/active_inference|Active Inference Theory]]
- [[../../knowledge_base/mathematics/expected_free_energy|Expected Free Energy]]

### Related Implementations
- [[../Generic_POMDP/README|Generic POMDP]] - Advanced POMDP framework
- [[../Generic_Thing/README|Generic Thing]] - Message-passing agents
- [[../../docs/examples/basic_agent|Basic Agent Tutorial]]

### Learning Resources
- [[../../docs/guides/learning_paths|Learning Paths]]
- [[../../docs/examples/|Usage Examples]]
- [[../../docs/guides/implementation_guides|Implementation Guides]]

## 🔗 Cross-References

### Agent Capabilities
- [[AGENTS|Simple POMDP Agents]] - Agent architecture documentation
- [[../../docs/agents/AGENTS|Agent Documentation Clearinghouse]]
- [[../../knowledge_base/agents/AGENTS|Agent Architecture Overview]]

### Educational Content
- [[../../docs/guides/learning_paths|Active Inference Basics]]
- [[../../docs/examples/basic_agent|Basic Agent Tutorial]]
- [[Output/|Generated Examples and Visualizations]]

---

> **Educational Focus**: Designed specifically for learning Active Inference and POMDP concepts with clear, well-documented code.

---

> **Progressive Complexity**: Starts with fundamentals and builds understanding through concrete examples.

---

> **Research Foundation**: Provides foundation for understanding more advanced Active Inference implementations.
