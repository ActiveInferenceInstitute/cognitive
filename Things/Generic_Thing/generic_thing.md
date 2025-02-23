# Generic Thing

A universal building block that implements Active Inference principles for creating adaptive, intelligent systems. The Generic Thing provides a foundational framework for building complex, hierarchical systems that learn and adapt through the Free Energy Principle.

## Core Principles

This implementation rigorously follows:

- **Active Inference**: A framework for understanding adaptive systems through the lens of free energy minimization
- **Free Energy Principle**: Systems act to minimize their variational free energy, which bounds surprise
- **Message Passing**: Efficient belief propagation between components using asynchronous queues
- **Federated Inference**: Distributed inference across multiple interacting things
- **Nested Markov Blankets**: Hierarchical organization of conditional independence relationships
- **Holographic Holons**: Each part contains aspects of the whole while being part of larger wholes

## Architecture

The Generic Thing follows a modular architecture with these key components:

### Core Components

1. [[GenericThing]] (`core.py`)
   - Main class implementing the generic thing interface
   - Manages internal state, external state, and actions
   - Coordinates between other components
   - Supports hierarchical nesting of things
   - Implements state observation and action generation

2. [[MarkovBlanket]] (`markov_blanket.py`)
   - Implements the Markov blanket concept
   - Manages sensory states, active states, and internal states
   - Maintains conditional independence relationships
   - Handles state updates and transitions
   - Implements boundary conditions and interface management

3. [[FreeEnergy]] (`free_energy.py`)
   - Implements the Free Energy Principle
   - Computes and minimizes variational free energy
   - Handles perception and action selection
   - Maintains generative models and recognition densities
   - Implements prediction error minimization

4. [[MessagePassing]] (`message_passing.py`)
   - Manages asynchronous communication between things
   - Implements belief propagation algorithms
   - Handles message queues and routing
   - Maintains connection topology
   - Supports different message types and priorities

5. [[FederatedInference]] (`inference.py`)
   - Implements distributed inference across things
   - Handles belief updates and consensus formation
   - Manages evidence accumulation
   - Supports parameter sharing
   - Implements distributed learning algorithms

### Visualization & Analysis

The package includes comprehensive visualization tools (`visualization.py`):
- Network topology visualization
- State evolution plots
- Free energy landscapes
- Belief distribution visualization
- Interactive debugging tools

## Implementation Details

### State Management

Things maintain several types of state:
```python
class GenericThing:
    internal_state: Dict[str, Any]  # Internal beliefs and parameters
    external_state: Dict[str, Any]  # Perceived external environment
    markov_blanket: MarkovBlanket   # Interface with other things
```

### Message Types

The system supports various message types:
1. Observations
2. Actions
3. Beliefs
4. Parameters
5. Control signals

### Update Cycle

Each thing follows this update cycle:
1. Receive observations
2. Update Markov blanket
3. Minimize free energy
4. Generate actions
5. Propagate messages

## Usage Examples

### 1. Basic Thing Creation
```python
from generic_thing import GenericThing

# Create a basic thing
thing = GenericThing(
    id="thing1",
    name="Example Thing",
    description="A simple example thing"
)

# Update with observations
thing.update({
    "sensor_1": 0.5,
    "sensor_2": [1.0, 2.0]
})

# Generate actions
actions = thing.act()
```

### 2. Hierarchical Composition
```python
# Create parent and child things
parent = GenericThing(id="parent", name="Parent Thing")
child1 = GenericThing(id="child1", name="Child Thing 1")
child2 = GenericThing(id="child2", name="Child Thing 2")

# Build hierarchy
parent.add_child(child1)
parent.add_child(child2)

# Get complete state
state = parent.get_state()  # Includes all children states
```

### 3. Message Passing Network
```python
# Connect things
thing1.message_passing.connect(thing2.id)
thing2.message_passing.connect(thing3.id)

# Send messages
thing1.update({"observation": "data"})  # Automatically propagates
```

### 4. Free Energy Minimization
```python
# Define prior beliefs
thing.free_energy.set_prior({
    "expected_state": np.array([0.0, 1.0]),
    "precision": np.array([[1.0, 0.0], [0.0, 1.0]])
})

# Update beliefs and minimize free energy
thing.update({"new_observation": [0.1, 0.9]})
```

## Testing

The package includes comprehensive test suites:

1. Unit Tests
   - `test_core.py`: Core functionality tests
   - `test_markov_blanket.py`: Boundary tests
   - `test_free_energy.py`: Energy minimization tests
   - `test_message_passing.py`: Communication tests
   - `test_inference.py`: Inference mechanism tests

2. Integration Tests
   - `test_thing_interactions.py`: Multi-thing interaction tests
   - `test_repertoires.py`: Behavioral repertoire tests

3. Performance Tests
   - `run_benchmarks.py`: Performance benchmarking
   - `run_test_suite.py`: Complete test suite runner

Run tests using:
```bash
python -m pytest
```

## Dependencies

Core dependencies (see `requirements.txt` for versions):
- numpy: Numerical computations
- matplotlib: Visualization
- seaborn: Advanced plotting
- networkx: Graph operations
- pytest: Testing framework

## Extension Points

The Generic Thing can be extended through:

1. Subclassing Core Components
```python
class SpecializedThing(GenericThing):
    def custom_update(self):
        # Specialized update logic
        pass
```

2. Custom State Definitions
3. Modified Update Rules
4. Specialized Message Types
5. Custom Inference Mechanisms

## Future Directions

1. **Enhanced Learning**
   - Deep learning integration
   - Meta-learning capabilities
   - Online learning algorithms

2. **Optimization**
   - Parallel processing
   - GPU acceleration
   - Distributed computing

3. **Visualization**
   - Real-time monitoring
   - 3D visualization
   - Interactive dashboards

4. **Integration**
   - REST API interface
   - Event streaming
   - Database connectivity

5. **Applications**
   - Robotics control
   - Multi-agent systems
   - Cognitive architectures

## Contributing

See `CONTRIBUTING.md` for guidelines on:
- Code style
- Testing requirements
- Documentation standards
- Pull request process

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 