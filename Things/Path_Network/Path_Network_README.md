# Path Network: Active Inference Agents in a Dynamic Topology

## Project Overview
We want a simulation where an arbitrary topology (randomly initially generated) of nodes, reflecting their communicative effective causal mutual influence, is generated and simulated. Each node is a Continuous time, Active Inference agent.

Different nodes can have different relations with the same underlying inference methods, for example different time delta of integration, different levels of Taylor Series expansion for generalized coordinate anticipatory inference.

At a big picture level, it is like there are nested Sin waves that are the "level of water in the world". All of the nodes are like ships or towers, that are doing this real-time inference on how much to ascend or descend in order to stay within a tolerable limited range of the overall sea level (which floats all boats).

## Project Structure

```
path_network/
├── core/
│   ├── __init__.py
│   ├── agent.py              # Active Inference agent implementation
│   ├── network.py            # Network topology and communication
│   ├── dynamics.py           # Sin wave and environmental dynamics
│   └── inference.py          # Generalized coordinates and inference methods
├── utils/
│   ├── __init__.py
│   ├── visualization.py      # Network and dynamics visualization
│   └── math_utils.py         # Mathematical utilities
├── config/
│   ├── __init__.py
│   └── settings.py           # Simulation parameters
└── simulation/
    ├── __init__.py
    └── runner.py             # Main simulation orchestration
```

## Core Components

### 1. Active Inference Agent (agent.py)
- Continuous-time state estimation
- Free energy minimization
- Generalized coordinates for anticipatory inference
- Adaptive response mechanisms
- Individual parameter settings for:
  - Integration time delta
  - Taylor series expansion levels
  - Tolerance ranges
  - Response dynamics

### 2. Network Topology (network.py)
- Random topology generation
- Inter-agent communication protocols
- Influence weight matrices
- Dynamic topology updates
- Neighborhood relationships

### 3. Environmental Dynamics (dynamics.py)
- Nested sinusoidal wave generation
- Global state management
- Environmental perturbations
- Time evolution systems

### 4. Inference Methods (inference.py)
- Generalized coordinates implementation
- Free energy computation
- Variational inference algorithms
- Prediction error minimization

## Implementation Plan

1. **Phase 1: Core Infrastructure**
   - Set up project structure
   - Implement basic mathematical utilities
   - Create configuration management
   - Develop testing framework

2. **Phase 2: Agent Implementation**
   - Implement core Active Inference mathematics
   - Create base agent class
   - Add generalized coordinate systems
   - Implement free energy minimization

3. **Phase 3: Network Development**
   - Create topology generation
   - Implement inter-agent communication
   - Develop influence propagation
   - Add dynamic network updates

4. **Phase 4: Environmental Systems**
   - Implement sinusoidal wave systems
   - Create global state management
   - Add perturbation mechanisms
   - Develop state tracking

5. **Phase 5: Visualization and Analysis**
   - Create network visualization
   - Implement state space plotting
   - Add analysis tools
   - Create performance metrics

6. **Phase 6: Integration and Testing**
   - Integrate all components
   - Comprehensive testing
   - Performance optimization
   - Documentation

## Key Features

1. **Dynamic Topology**
   - Random initial generation
   - Adaptive connectivity
   - Influence-based relationships
   - Real-time updates

2. **Active Inference Implementation**
   - Continuous time processing
   - Generalized coordinate inference
   - Variable integration time steps
   - Adaptive response mechanisms

3. **Environmental Simulation**
   - Multi-frequency sin waves
   - Global state management
   - Perturbation systems
   - State tracking

4. **Visualization and Analysis**
   - Network topology visualization
   - State space plotting
   - Performance metrics
   - Real-time monitoring

## Dependencies
- NumPy: Numerical computations
- NetworkX: Graph topology management
- SciPy: Scientific computations
- Matplotlib: Visualization
- PyTorch: Deep learning and optimization
- Pandas: Data management

## Getting Started
[Implementation to follow] 