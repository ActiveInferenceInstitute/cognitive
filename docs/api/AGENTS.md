---
title: API Agent Documentation
type: agents
status: stable
created: 2025-01-01
updated: 2025-01-01
tags:
  - api
  - agents
  - documentation
  - interfaces
semantic_relations:
  - type: documents
    links:
      - [[api_documentation]]
      - [[../../tools/src/models/active_inference/AGENTS]]
---

# API Agent Documentation

Technical API documentation for agent interfaces, classes, and methods within the Active Inference cognitive modeling framework. This documentation provides comprehensive reference information for developers implementing and interacting with cognitive agents.

## 🔌 Agent API Overview

### Core Agent Interfaces

#### Base Agent Interface
Fundamental interface that all cognitive agents implement, providing standardized interaction patterns.

```python
class BaseAgentInterface:
    """Abstract base class defining the core agent interface."""

    @abstractmethod
    def __init__(self, config: dict):
        """Initialize agent with configuration parameters."""
        pass

    @abstractmethod
    def perceive(self, observation: np.ndarray) -> dict:
        """Process sensory input and update internal state.

        Args:
            observation: Sensory input data

        Returns:
            dict: Perception results including beliefs and predictions
        """
        pass

    @abstractmethod
    def act(self, context: dict) -> dict:
        """Generate action based on current state and goals.

        Args:
            context: Current situational context

        Returns:
            dict: Action specification and associated metadata
        """
        pass

    @abstractmethod
    def learn(self, experience: dict) -> dict:
        """Update agent knowledge based on experience.

        Args:
            experience: Experience data (observations, actions, outcomes)

        Returns:
            dict: Learning results and parameter updates
        """
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """Get current agent internal state.

        Returns:
            dict: Complete agent state representation
        """
        pass
```

#### Active Inference Agent Interface
Specialized interface for Active Inference agents with variational inference capabilities.

```python
class ActiveInferenceAgentInterface(BaseAgentInterface):
    """Extended interface for Active Inference agents."""

    @abstractmethod
    def compute_free_energy(self, beliefs: dict, observations: np.ndarray) -> float:
        """Compute variational free energy for current state.

        Args:
            beliefs: Current belief distribution
            observations: Current observations

        Returns:
            float: Variational free energy value
        """
        pass

    @abstractmethod
    def update_beliefs(self, observations: np.ndarray, actions: np.ndarray) -> dict:
        """Update beliefs using variational inference.

        Args:
            observations: New observation data
            actions: Recent action history

        Returns:
            dict: Updated belief distributions
        """
        pass

    @abstractmethod
    def select_policy(self, beliefs: dict, goals: dict) -> dict:
        """Select optimal policy using Expected Free Energy minimization.

        Args:
            beliefs: Current belief state
            goals: Goal specifications

        Returns:
            dict: Selected policy and expected outcomes
        """
        pass
```

## 🏗️ Agent Class Hierarchy

### Core Agent Classes

#### GenericThing
Fundamental Active Inference agent providing message-passing cognition.

```python
class GenericThing(ActiveInferenceAgentInterface):
    """Core Active Inference agent with message-passing architecture.

    Attributes:
        markov_blanket: Boundary between internal and external states
        free_energy_system: Variational free energy computation
        message_passer: Inter-agent communication system
        inference_engine: Federated inference coordination
    """

    def __init__(self, config: dict):
        """Initialize Generic Thing agent.

        Args:
            config: Agent configuration dictionary containing:
                - state_space_size: Size of state space
                - message_buffer_size: Communication buffer capacity
                - inference_precision: Variational inference precision
                - learning_rate: Parameter learning rate
        """
        super().__init__(config)
        self.markov_blanket = MarkovBlanket(config)
        self.free_energy_system = FreeEnergyCalculator(config)
        self.message_passer = MessagePassingSystem(config)
        self.inference_engine = FederatedInferenceEngine(config)

    def perceive_and_act(self, sensory_input: dict) -> dict:
        """Complete perception-action cycle.

        Args:
            sensory_input: Multi-modal sensory data

        Returns:
            dict: Action specification and internal state updates
        """
        # Implementation details...
        pass
```

#### POMDP Agents
Partially observable Markov decision process agents with belief tracking.

```python
class GenericPOMDP(ActiveInferenceAgentInterface):
    """Advanced POMDP agent with hierarchical planning.

    Attributes:
        A_matrix: Observation model P(o|s)
        B_matrix: Transition model P(s'|s,a)
        C_matrix: Preferences over observations
        D_matrix: Initial belief distribution
        E_matrix: Action prior preferences
    """

    def __init__(self, config: dict):
        """Initialize POMDP agent.

        Args:
            config: Configuration with matrix dimensions and parameters:
                - num_observations: Size of observation space
                - num_states: Size of state space
                - num_actions: Size of action space
                - planning_horizon: Temporal planning depth
        """
        super().__init__(config)
        self.A_matrix = self._initialize_A_matrix(config)
        self.B_matrix = self._initialize_B_matrix(config)
        self.C_matrix = self._initialize_C_matrix(config)
        self.D_matrix = self._initialize_D_matrix(config)
        self.E_matrix = self._initialize_E_matrix(config)

    def step(self, observation: int = None, action: int = None) -> tuple:
        """Execute single POMDP step.

        Args:
            observation: Current observation (optional)
            action: Action to execute (optional)

        Returns:
            tuple: (observation, free_energy)
        """
        # Implementation details...
        pass
```

### Specialized Agent Classes

#### Continuous Agents
Agents operating in continuous state and action spaces.

```python
class ContinuousGenericAgent(BaseAgentInterface):
    """Agent for continuous-time Active Inference.

    Attributes:
        state_dynamics: Continuous state evolution equations
        observation_model: Continuous observation mapping
        control_system: Continuous action generation
    """

    def __init__(self, config: dict):
        """Initialize continuous agent.

        Args:
            config: Continuous agent configuration:
                - state_dimension: Dimensionality of state space
                - observation_dimension: Dimensionality of observation space
                - time_step: Integration time step
                - integration_method: Numerical integration method
        """
        super().__init__(config)
        self.state_dynamics = ContinuousDynamics(config)
        self.observation_model = ContinuousObservationModel(config)
        self.control_system = ContinuousControlSystem(config)

    def continuous_update(self, dt: float) -> np.ndarray:
        """Continuous-time update cycle.

        Args:
            dt: Time step for integration

        Returns:
            np.ndarray: Continuous action vector
        """
        # Implementation details...
        pass
```

#### Swarm Agents
Multi-agent systems with collective intelligence.

```python
class SwarmAgent(BaseAgentInterface):
    """Agent participating in swarm intelligence systems.

    Attributes:
        local_perception: Individual sensory capabilities
        communication_system: Inter-agent communication
        coordination_mechanisms: Swarm coordination algorithms
        collective_memory: Shared knowledge repository
    """

    def __init__(self, config: dict):
        """Initialize swarm agent.

        Args:
            config: Swarm configuration:
                - swarm_size: Number of agents in swarm
                - communication_range: Communication radius
                - coordination_algorithm: Swarm coordination method
        """
        super().__init__(config)
        self.local_perception = LocalPerceptionSystem(config)
        self.communication_system = InterAgentCommunication(config)
        self.coordination_mechanisms = SwarmCoordination(config)
        self.collective_memory = SharedKnowledgeBase(config)
```

## 🔧 Agent Configuration API

### Configuration Classes

#### AgentConfig
Configuration management for agent initialization and parameters.

```python
@dataclass
class AgentConfig:
    """Configuration class for agent parameters."""

    # Core agent parameters
    agent_type: str = "GenericThing"
    state_space_size: int = 10
    action_space_size: int = 3
    observation_space_size: int = 5

    # Learning parameters
    learning_rate: float = 0.01
    momentum: float = 0.9
    precision: float = 1.0

    # Planning parameters
    planning_horizon: int = 5
    discount_factor: float = 0.99

    # System parameters
    random_seed: Optional[int] = None
    logging_level: str = "INFO"
    performance_monitoring: bool = True

    def validate(self) -> bool:
        """Validate configuration parameters."""
        # Implementation details...
        pass

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)
```

#### EnvironmentConfig
Configuration for agent-environment interactions.

```python
@dataclass
class EnvironmentConfig:
    """Configuration for agent operating environments."""

    environment_type: str = "GridWorld"
    dimensions: Tuple[int, int] = (10, 10)
    obstacle_density: float = 0.1

    reward_structure: dict = field(default_factory=lambda: {
        "goal": 1.0,
        "obstacle": -1.0,
        "step": -0.01
    })

    dynamics: dict = field(default_factory=lambda: {
        "deterministic": False,
        "noise_level": 0.1,
        "time_limit": 1000
    })

    observation_model: dict = field(default_factory=lambda: {
        "modality": "vision",
        "resolution": [64, 64],
        "noise": 0.05
    })
```

## 📊 Agent State and Monitoring API

### State Management

#### AgentState
Comprehensive agent state representation.

```python
@dataclass
class AgentState:
    """Complete representation of agent internal state."""

    # Belief distributions
    beliefs: np.ndarray
    belief_history: List[np.ndarray]

    # Policy information
    current_policy: dict
    policy_history: List[dict]

    # Performance metrics
    free_energy: float
    free_energy_history: List[float]

    # Learning state
    parameters: dict
    parameter_history: List[dict]

    # System state
    timestamp: float
    iteration_count: int

    def to_dict(self) -> dict:
        """Convert state to serializable dictionary."""
        # Implementation details...
        pass

    def save(self, filepath: str):
        """Save agent state to file."""
        # Implementation details...
        pass

    @classmethod
    def load(cls, filepath: str) -> 'AgentState':
        """Load agent state from file."""
        # Implementation details...
        pass
```

### Performance Monitoring

#### AgentMonitor
Real-time performance tracking and analysis.

```python
class AgentMonitor:
    """Real-time agent performance monitoring and analysis."""

    def __init__(self, agent: BaseAgentInterface, config: dict):
        """Initialize agent monitor.

        Args:
            agent: Agent instance to monitor
            config: Monitoring configuration
        """
        self.agent = agent
        self.config = config
        self.metrics_history = defaultdict(list)
        self.alerts = []

    def track_performance(self):
        """Track current agent performance metrics."""
        # Implementation details...
        pass

    def get_metrics(self) -> dict:
        """Get current performance metrics."""
        # Implementation details...
        pass

    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        # Implementation details...
        pass

    def check_alerts(self) -> List[dict]:
        """Check for performance alerts."""
        # Implementation details...
        pass
```

## 🚀 Agent Factory and Management API

### AgentFactory
Factory pattern for agent creation and configuration.

```python
class AgentFactory:
    """Factory for creating and configuring agents."""

    @staticmethod
    def create_agent(agent_type: str, config: dict) -> BaseAgentInterface:
        """Create agent instance based on type and configuration.

        Args:
            agent_type: Type of agent to create
            config: Agent configuration dictionary

        Returns:
            BaseAgentInterface: Configured agent instance

        Raises:
            ValueError: If agent type is not supported
        """
        agent_classes = {
            "GenericThing": GenericThing,
            "GenericPOMDP": GenericPOMDP,
            "ContinuousGeneric": ContinuousGenericAgent,
            "SwarmAgent": SwarmAgent,
        }

        if agent_type not in agent_classes:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        agent_class = agent_classes[agent_type]
        return agent_class(config)

    @staticmethod
    def create_environment(env_type: str, config: dict):
        """Create environment instance."""
        # Implementation details...
        pass

    @classmethod
    def create_agent_environment_pair(cls, agent_config: dict, env_config: dict):
        """Create matched agent and environment pair."""
        # Implementation details...
        pass
```

### AgentManager
High-level agent lifecycle and orchestration management.

```python
class AgentManager:
    """High-level agent lifecycle and orchestration management."""

    def __init__(self):
        """Initialize agent manager."""
        self.active_agents = {}
        self.environments = {}
        self.monitors = {}

    def create_agent(self, agent_id: str, agent_type: str, config: dict) -> str:
        """Create and register new agent.

        Args:
            agent_id: Unique identifier for agent
            agent_type: Type of agent to create
            config: Agent configuration

        Returns:
            str: Agent identifier
        """
        # Implementation details...
        pass

    def start_simulation(self, agent_id: str, env_id: str, duration: int):
        """Start agent-environment simulation."""
        # Implementation details...
        pass

    def stop_simulation(self, agent_id: str):
        """Stop agent simulation."""
        # Implementation details...
        pass

    def get_agent_status(self, agent_id: str) -> dict:
        """Get current agent status and metrics."""
        # Implementation details...
        pass
```

## 📚 API Documentation

### Complete API Reference
- **Class Documentation**: Detailed class and method documentation
- **Parameter Specifications**: Complete parameter descriptions and types
- **Return Value Documentation**: Detailed return value specifications
- **Exception Documentation**: Error conditions and exception handling

### Usage Examples
- **Basic Agent Creation**: Simple agent instantiation examples
- **Configuration Examples**: Complete configuration examples
- **Integration Examples**: System integration patterns
- **Advanced Usage**: Complex multi-agent scenarios

### Code Examples
```python
# Create and configure agent
config = {
    "agent_type": "GenericPOMDP",
    "num_states": 10,
    "num_observations": 5,
    "num_actions": 3,
    "planning_horizon": 4
}

agent = AgentFactory.create_agent("GenericPOMDP", config)

# Run simulation with monitoring
monitor = AgentMonitor(agent, {"metrics": ["free_energy", "policy_entropy"]})

for step in range(100):
    observation = environment.get_observation()
    action = agent.select_action(observation)
    reward = environment.step(action)
    agent.learn(reward)

    monitor.track_performance()

# Generate performance report
report = monitor.generate_report()
print(report)
```

## 🔗 Related Documentation

### Implementation References
- [[../../tools/src/models/active_inference/|Active Inference Models]]
- [[../../Things/Generic_Thing/README|Generic Thing Implementation]]
- [[../../Things/Generic_POMDP/README|Generic POMDP Implementation]]

### Configuration Resources
- [[../config/README|Configuration Documentation]]
- [[simulation_config|Simulation Configuration]]
- [[../../tools/src/utils/config|Configuration Utilities]]

### Testing and Validation
- [[../../tests/README|Testing Framework]]
- [[../../tests/test_api|API Tests]]
- [[../repo_docs/api_testing|API Testing Guidelines]]

## 🔗 Cross-References

### Agent Types
- [[../../Things/Generic_Thing/AGENTS|Generic Thing Agents]]
- [[../../Things/Simple_POMDP/AGENTS|Simple POMDP Agents]]
- [[../../Things/Generic_POMDP/AGENTS|Generic POMDP Agents]]
- [[../../Things/Continuous_Generic/AGENTS|Continuous Generic Agents]]

### API Components
- **Agent Interfaces**: Core agent interaction contracts
- **Configuration Classes**: Agent and environment configuration
- **Factory Classes**: Agent creation and management utilities
- **Monitoring Classes**: Performance tracking and analysis

---

> **Comprehensive API**: Complete technical reference for all agent classes, interfaces, and interaction patterns in the framework.

---

> **Developer Focused**: Detailed technical documentation designed for framework developers and advanced users.

---

> **Integration Ready**: Well-documented APIs enabling seamless integration with external systems and applications.
