# Ant Colony - Active Inference Multi-Agent System

## Overview
This implementation models an ant colony as a multi-agent system where each ant (Nestmate) operates according to the Free Energy Principle (FEP). The system demonstrates how individual agents following active inference can give rise to emergent collective behaviors and self-organization.

## Core Components

### 1. Nestmate Agent
Individual ant agents that implement active inference:
- Sensory observations (pheromones, food, obstacles, nestmates)
- Internal generative model of environment
- Action selection through FEP
- Belief updating and learning
- Pheromone deposition behaviors

### 2. Colony Environment
Shared environment where agents interact:
- Pheromone diffusion and evaporation
- Food source dynamics
- Obstacle and terrain features
- Nest structure and organization
- Physical constraints and interactions

### 3. Multi-Agent Framework
System for managing agent interactions:
- Agent communication protocols
- Spatial relationships
- Task allocation
- Resource distribution
- Collective decision making

## Implementation Details

### Nestmate Agent Model
```python
class Nestmate:
    def __init__(self):
        # State space
        self.position = None        # Physical location
        self.orientation = None     # Heading direction
        self.carrying = None        # What agent is carrying
        self.energy = 1.0          # Energy level
        
        # Sensory inputs
        self.observations = {
            'pheromone': None,     # Pheromone gradients
            'food': None,          # Food sources
            'nestmates': None,     # Other agents
            'obstacles': None       # Environmental obstacles
        }
        
        # Internal model parameters
        self.beliefs = None        # Current belief state
        self.preferences = None    # Goal-directed preferences
        self.policies = None       # Action policies
        
        # Learning parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
```

### Key Features

1. Active Inference Implementation
- Hierarchical generative models
- Precision-weighted prediction errors
- Free energy minimization
- Action-perception cycles
- Belief updating through message passing

2. Pheromone System
- Multiple pheromone types
- Diffusion mechanics
- Evaporation rates
- Gradient following
- Trail reinforcement

3. Task Allocation
- Foraging
- Nest maintenance
- Brood care
- Defense
- Exploration

4. Learning & Adaptation
- Individual learning
- Social learning
- Environmental adaptation
- Task switching
- Skill development

## Configuration

The system is configured through YAML files:

1. agent_config.yaml - Individual agent parameters
2. colony_config.yaml - Colony-level parameters
3. environment_config.yaml - Environmental settings
4. simulation_config.yaml - Simulation parameters

## Usage

```python
from ant_colony import Colony, Environment, Simulation

# Initialize environment
env = Environment(config_path='config/environment_config.yaml')

# Create colony
colony = Colony(
    num_agents=100,
    environment=env,
    config_path='config/colony_config.yaml'
)

# Run simulation
sim = Simulation(
    colony=colony,
    config_path='config/simulation_config.yaml'
)
sim.run(timesteps=1000)
```

## Analysis Tools

1. Behavioral Analysis
- Task distribution metrics
- Spatial patterns
- Temporal dynamics
- Efficiency measures

2. Network Analysis
- Interaction networks
- Information flow
- Task networks
- Spatial networks

3. Collective Intelligence Metrics
- Emergence measures
- Synchronization indices
- Collective decision accuracy
- Adaptation rates

## Visualization

1. Real-time Visualization
- Agent positions and movements
- Pheromone gradients
- Resource distribution
- Task allocation

2. Analysis Plots
- Behavioral statistics
- Learning curves
- Network diagrams
- Performance metrics

## Project Structure
```
ant_colony/
├── agents/
│   ├── nestmate.py
│   ├── sensor.py
│   └── actuator.py
├── environment/
│   ├── world.py
│   ├── pheromone.py
│   └── resources.py
├── models/
│   ├── generative_model.py
│   ├── belief_update.py
│   └── policy_selection.py
├── analysis/
│   ├── metrics.py
│   ├── network.py
│   └── visualization.py
├── config/
│   ├── agent_config.yaml
│   ├── colony_config.yaml
│   ├── environment_config.yaml
│   └── simulation_config.yaml
└── tests/
    ├── test_agent.py
    ├── test_environment.py
    └── test_simulation.py
```
