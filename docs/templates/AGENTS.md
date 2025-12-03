---
title: Templates Agent Documentation
type: agents
status: stable
created: 2025-01-01
updated: 2025-01-01
tags:
  - templates
  - agents
  - reusable
  - patterns
semantic_relations:
  - type: templates
    links:
      - [[agent_template]]
      - [[../../docs/guides/AGENTS]]
---

# Templates Agent Documentation

Reusable agent templates and patterns for rapid development and consistent implementation of cognitive agents within the Active Inference framework. These templates provide standardized structures, best practices, and extensible foundations for building reliable, maintainable agent systems.

## 📋 Agent Template Categories

### Basic Agent Templates

#### [[agent_template|Basic Agent Template]]
Fundamental template for creating simple Active Inference agents with essential components.

```python
# File: templates/basic_agent_template.py

from cognitive_modeling.agents import BaseAgent
import numpy as np

class BasicAgentTemplate(BaseAgent):
    """
    Basic Active Inference agent template.

    This template provides the fundamental structure for implementing
    a simple Active Inference agent with belief updating, action selection,
    and learning capabilities.

    Attributes:
        beliefs (np.ndarray): Current belief distribution over states
        learning_rate (float): Rate of belief updating
        action_preferences (np.ndarray): Preferences over actions
        state_history (list): History of belief states
    """

    def __init__(self, config: dict):
        """
        Initialize the basic agent template.

        Args:
            config (dict): Configuration dictionary containing:
                - state_space_size (int): Number of possible states
                - action_space_size (int): Number of possible actions
                - learning_rate (float): Learning rate for updates
                - initial_beliefs (np.ndarray, optional): Initial belief distribution
        """
        super().__init__(config)

        # Extract configuration parameters
        self.state_space_size = config.get('state_space_size', 5)
        self.action_space_size = config.get('action_space_size', 3)
        self.learning_rate = config.get('learning_rate', 0.01)

        # Initialize belief state
        if 'initial_beliefs' in config:
            self.beliefs = np.array(config['initial_beliefs'])
        else:
            # Uniform initial beliefs
            self.beliefs = np.ones(self.state_space_size) / self.state_space_size

        # Initialize action preferences
        self.action_preferences = np.zeros((self.state_space_size, self.action_space_size))

        # Initialize history tracking
        self.state_history = [self.beliefs.copy()]
        self.action_history = []

        # Validate configuration
        self._validate_configuration()

    def perceive(self, observation: np.ndarray) -> dict:
        """
        Update beliefs based on sensory observation.

        This method implements the perception aspect of Active Inference,
        updating the agent's belief state based on new sensory information.

        Args:
            observation (np.ndarray): Sensory observation data

        Returns:
            dict: Perception results containing:
                - beliefs: Updated belief distribution
                - prediction_error: Prediction error magnitude
                - confidence: Confidence in current beliefs
        """
        # Compute likelihood of observation given current beliefs
        likelihood = self._compute_likelihood(observation)

        # Update beliefs using Bayes rule (simplified)
        posterior = self.beliefs * likelihood
        posterior = posterior / np.sum(posterior)  # Normalize

        # Apply learning rate
        self.beliefs = (1 - self.learning_rate) * self.beliefs + self.learning_rate * posterior

        # Ensure beliefs remain valid probability distribution
        self.beliefs = np.clip(self.beliefs, 1e-6, 1.0)
        self.beliefs = self.beliefs / np.sum(self.beliefs)

        # Track state history
        self.state_history.append(self.beliefs.copy())

        # Compute prediction error and confidence
        prediction_error = np.linalg.norm(posterior - self.beliefs)
        confidence = 1.0 / (1.0 + prediction_error)  # Higher error = lower confidence

        return {
            'beliefs': self.beliefs.copy(),
            'prediction_error': prediction_error,
            'confidence': confidence
        }

    def act(self, context: dict = None) -> dict:
        """
        Select action based on current beliefs and preferences.

        This method implements the action selection aspect of Active Inference,
        choosing actions that minimize expected free energy.

        Args:
            context (dict, optional): Additional context information

        Returns:
            dict: Action selection results containing:
                - action: Selected action index
                - action_values: Values for all possible actions
                - expected_free_energy: EFE for selected action
        """
        # Compute expected free energy for each action
        action_values = self._compute_expected_free_energy()

        # Select action with minimum EFE (greedy selection)
        selected_action = np.argmin(action_values)

        # Track action history
        self.action_history.append(selected_action)

        return {
            'action': selected_action,
            'action_values': action_values,
            'expected_free_energy': action_values[selected_action]
        }

    def learn(self, experience: dict) -> dict:
        """
        Update agent knowledge based on experience.

        This method implements the learning aspect of Active Inference,
        updating action preferences and other parameters based on experience.

        Args:
            experience (dict): Experience data containing:
                - observation: Sensory observation
                - action: Action taken
                - reward: Reward received
                - next_observation: Subsequent observation

        Returns:
            dict: Learning results containing:
                - parameter_updates: Changes made to parameters
                - learning_progress: Learning metrics
        """
        observation = experience['observation']
        action = experience['action']
        reward = experience['reward']

        # Update action preferences based on reward
        # Simple Q-learning style update
        preference_update = self.learning_rate * (reward - self.action_preferences[:, action].max())
        self.action_preferences[:, action] += preference_update

        return {
            'parameter_updates': {'action_preferences': preference_update},
            'learning_progress': {'reward': reward}
        }

    def _compute_likelihood(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute likelihood of observation given each state.

        This is a placeholder implementation - override in subclasses
        for specific observation models.

        Args:
            observation (np.ndarray): Sensory observation

        Returns:
            np.ndarray: Likelihood for each state
        """
        # Simple placeholder: uniform likelihood
        return np.ones(self.state_space_size) / self.state_space_size

    def _compute_expected_free_energy(self) -> np.ndarray:
        """
        Compute expected free energy for each action.

        This implements a simplified EFE computation based on current beliefs
        and action preferences.

        Returns:
            np.ndarray: Expected free energy for each action
        """
        efe_values = np.zeros(self.action_space_size)

        for action in range(self.action_space_size):
            # Simplified EFE: negative preference + exploration bonus
            preference_term = -np.dot(self.beliefs, self.action_preferences[:, action])

            # Exploration bonus based on action uncertainty
            uncertainty_bonus = 0.1 * np.sqrt(np.var(self.action_preferences[:, action]))

            efe_values[action] = preference_term - uncertainty_bonus

        return efe_values

    def _validate_configuration(self):
        """Validate configuration parameters."""
        if self.state_space_size <= 0:
            raise ValueError("state_space_size must be positive")
        if self.action_space_size <= 0:
            raise ValueError("action_space_size must be positive")
        if not (0 < self.learning_rate <= 1):
            raise ValueError("learning_rate must be in (0, 1]")

    def get_state(self) -> dict:
        """
        Get current agent state.

        Returns:
            dict: Complete agent state information
        """
        return {
            'beliefs': self.beliefs.copy(),
            'action_preferences': self.action_preferences.copy(),
            'state_history': self.state_history.copy(),
            'action_history': self.action_history.copy(),
            'config': {
                'state_space_size': self.state_space_size,
                'action_space_size': self.action_space_size,
                'learning_rate': self.learning_rate
            }
        }

    def reset(self):
        """Reset agent to initial state."""
        self.beliefs = np.ones(self.state_space_size) / self.state_space_size
        self.action_preferences = np.zeros((self.state_space_size, self.action_space_size))
        self.state_history = [self.beliefs.copy()]
        self.action_history = []

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (f"BasicAgentTemplate("
                f"states={self.state_space_size}, "
                f"actions={self.action_space_size}, "
                f"lr={self.learning_rate:.3f})")
```

#### [[environment_template|Environment Template]]
Template for creating agent testing and training environments.

### Advanced Agent Templates

#### [[hierarchical_agent_template|Hierarchical Agent Template]]
Template for multi-level cognitive agents with hierarchical processing.

#### [[multi_agent_template|Multi-Agent Template]]
Template for coordinating multiple agents in shared environments.

### Domain-Specific Templates

#### [[robotics_agent_template|Robotics Agent Template]]
Specialized template for robotic control and navigation agents.

#### [[healthcare_agent_template|Healthcare Agent Template]]
Template for medical decision support and health monitoring agents.

#### [[finance_agent_template|Finance Agent Template]]
Template for financial decision-making and risk management agents.

## 🏗️ Template Architecture

### Template Structure
All templates follow a consistent architectural pattern:

```
templates/
├── base_templates/          # Fundamental agent templates
│   ├── basic_agent.py      # Basic Active Inference agent
│   ├── hierarchical_agent.py # Multi-level agent
│   └── multi_agent.py      # Multi-agent coordination
├── domain_templates/        # Domain-specific templates
│   ├── robotics/
│   ├── healthcare/
│   └── finance/
├── utility_templates/       # Utility and helper templates
│   ├── logging_agent.py    # Agent with logging capabilities
│   ├── visualization_agent.py # Agent with plotting features
│   └── testing_agent.py    # Agent with testing utilities
└── examples/                # Template usage examples
    ├── basic_usage.py      # Basic template usage
    ├── customization.py    # Template customization
    └── integration.py      # Template integration
```

### Template Components

#### Configuration Management
```python
@dataclass
class AgentConfig:
    """Configuration template for agent parameters."""

    # Core parameters
    name: str = "DefaultAgent"
    state_space_size: int = 10
    action_space_size: int = 3

    # Learning parameters
    learning_rate: float = 0.01
    discount_factor: float = 0.99
    exploration_rate: float = 0.1

    # Advanced parameters
    use_hierarchy: bool = False
    enable_learning: bool = True
    log_level: str = "INFO"

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        if self.state_space_size <= 0:
            errors.append("state_space_size must be positive")
        if not (0 <= self.learning_rate <= 1):
            errors.append("learning_rate must be in [0, 1]")
        return errors

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
```

#### Extension Mechanisms
```python
class ExtensibleAgentTemplate(BasicAgentTemplate):
    """Template demonstrating extension mechanisms."""

    def __init__(self, config: dict):
        super().__init__(config)

        # Extension registry
        self.extensions = {}

        # Load configured extensions
        self._load_extensions(config.get('extensions', []))

    def add_extension(self, name: str, extension):
        """Add functionality extension to agent."""
        self.extensions[name] = extension
        if hasattr(extension, 'on_attach'):
            extension.on_attach(self)

    def remove_extension(self, name: str):
        """Remove functionality extension."""
        if name in self.extensions:
            extension = self.extensions[name]
            if hasattr(extension, 'on_detach'):
                extension.on_detach(self)
            del self.extensions[name]

    def _load_extensions(self, extension_configs: List[dict]):
        """Load extensions from configuration."""
        for ext_config in extension_configs:
            ext_name = ext_config['name']
            ext_class = self._get_extension_class(ext_name)
            extension = ext_class(ext_config)
            self.add_extension(ext_name, extension)

    def _get_extension_class(self, name: str):
        """Get extension class by name."""
        # Extension registry - extend as needed
        extension_classes = {
            'logging': LoggingExtension,
            'visualization': VisualizationExtension,
            'learning': AdvancedLearningExtension,
        }
        return extension_classes.get(name)
```

## 🔧 Template Usage Patterns

### Basic Usage
```python
# Create agent from template
from templates.basic_agent_template import BasicAgentTemplate

config = {
    'state_space_size': 5,
    'action_space_size': 3,
    'learning_rate': 0.05
}

agent = BasicAgentTemplate(config)

# Use agent
for episode in range(100):
    observation = get_observation()
    action_result = agent.act()
    reward = environment.step(action_result['action'])
    agent.learn({'observation': observation, 'action': action_result['action'], 'reward': reward})
```

### Template Customization
```python
# Customize template for specific domain
class CustomDomainAgent(BasicAgentTemplate):
    """Customized agent for specific domain."""

    def _compute_likelihood(self, observation):
        """Domain-specific likelihood computation."""
        # Implement domain-specific observation model
        return domain_specific_likelihood(observation, self.beliefs)

    def _compute_expected_free_energy(self):
        """Domain-specific EFE computation."""
        # Implement domain-specific action evaluation
        return domain_specific_efe(self.beliefs, self.action_preferences)
```

### Template Composition
```python
# Compose multiple templates
class CompositeAgent(HierarchicalAgentTemplate, MultiAgentTemplate):
    """Agent combining hierarchical and multi-agent capabilities."""

    def __init__(self, config):
        # Initialize both template functionalities
        HierarchicalAgentTemplate.__init__(self, config)
        MultiAgentTemplate.__init__(self, config)

        # Additional composition logic
        self._resolve_template_conflicts()
```

## 📊 Template Quality Standards

### Code Quality
- **Documentation**: Comprehensive docstrings and comments
- **Type Hints**: Full type annotation coverage
- **Error Handling**: Robust error handling and validation
- **Testing**: Comprehensive test coverage

### Extensibility
- **Modular Design**: Easy to extend and customize
- **Configuration-Driven**: Behavior controlled by configuration
- **Hook Points**: Well-defined extension points
- **Composition**: Support for template composition

### Performance
- **Efficiency**: Optimized for computational performance
- **Scalability**: Scales to larger problem sizes
- **Memory Management**: Efficient memory usage
- **Real-time Capability**: Suitable for real-time applications

## 🎯 Template Categories

### By Complexity Level
- **Beginner Templates**: Simple, well-commented templates for learning
- **Intermediate Templates**: Feature-complete templates for development
- **Advanced Templates**: Research-grade templates with advanced features

### By Application Domain
- **General Purpose**: Broadly applicable agent templates
- **Domain Specific**: Specialized templates for particular applications
- **Research Oriented**: Templates designed for research and experimentation

### By Agent Architecture
- **Single Agent**: Individual agent templates
- **Multi-Agent**: Multiple agent coordination templates
- **Hierarchical**: Multi-level cognitive architecture templates

## 🔄 Template Maintenance

### Version Control
- **Semantic Versioning**: Clear versioning scheme for templates
- **Backward Compatibility**: Maintain compatibility across versions
- **Deprecation Policy**: Clear deprecation and migration path

### Update Process
- **Regular Updates**: Keep templates current with framework developments
- **Community Feedback**: Incorporate user feedback and suggestions
- **Performance Improvements**: Ongoing optimization and enhancement

### Quality Assurance
- **Automated Testing**: Comprehensive test suite for all templates
- **Code Review**: Peer review process for template changes
- **Usage Validation**: Validate templates with real-world usage

## 📚 Template Documentation

### Usage Guides
- **Getting Started**: Basic template usage instructions
- **Customization Guide**: How to customize and extend templates
- **Integration Guide**: Integrating templates with existing code
- **Best Practices**: Template usage best practices

### API Reference
- **Class Documentation**: Complete class and method documentation
- **Configuration Reference**: Detailed configuration parameter reference
- **Extension API**: Extension mechanism documentation

## 🔗 Related Documentation

### Implementation Resources
- [[../../docs/guides/AGENTS|Implementation Guides]]
- [[../../docs/examples/AGENTS|Examples Documentation]]
- [[../../tools/README|Development Tools]]

### Development Resources
- [[../development/README|Development Resources]]
- [[../../tests/README|Testing Framework]]

### Template Resources
- [[agent_template|Agent Template]]
- [[environment_template|Environment Template]]
- [[hierarchical_agent_template|Hierarchical Template]]

## 🔗 Cross-References

### Template Types
- **Basic Templates**: [[agent_template|Basic Agent]], [[environment_template|Environment]]
- **Advanced Templates**: [[hierarchical_agent_template|Hierarchical]], [[multi_agent_template|Multi-Agent]]
- **Domain Templates**: [[robotics_agent_template|Robotics]], [[healthcare_agent_template|Healthcare]]

### Usage Patterns
- **Basic Usage**: Standard template instantiation and usage
- **Customization**: Template extension and modification
- **Composition**: Combining multiple templates
- **Integration**: Template integration with existing systems

---

> **Rapid Development**: Standardized, reusable templates enabling rapid agent development with proven patterns and best practices.

---

> **Quality Assurance**: Thoroughly tested, well-documented templates ensuring reliable and maintainable agent implementations.

---

> **Extensibility**: Modular, configurable templates supporting easy customization and extension for diverse applications.
