# BioFirm Active Inference Model

A homeostatic control model using Active Inference principles for bioregion management. This implementation demonstrates how Active Inference can be used for maintaining system stability through partial observations.

## Overview

The BioFirm model implements a Partially Observable Markov Decision Process (POMDP) using Active Inference principles. The model aims to maintain a system in a homeostatic range by:

1. Making observations from a simplified 3-state observation space (Low/Medium/High)
2. Inferring the true underlying 5-state environment (Too Low, Lower Bound, Medium, Upper Bound, Too High)
3. Selecting actions (Decrease/Maintain/Increase) to keep the system in the desired state

## Model Components

### State Spaces
- **Observation Space**: 3 states
  - LOW (0)
  - MEDIUM (1)
  - HIGH (2)

- **Environment Space**: 5 states
  - TOO_LOW (0)
  - LOWER_BOUND (1)
  - MEDIUM (2)
  - UPPER_BOUND (3)
  - TOO_HIGH (4)

- **Action Space**: 3 actions
  - DECREASE (0)
  - MAINTAIN (1)
  - INCREASE (2)

### Active Inference Components

1. **Variational Free Energy (VFE)**
   - **Accuracy Term**: Measures how well beliefs explain observations
   - **Complexity Term**: KL divergence from prior beliefs
   - **Total VFE**: Accuracy + Complexity, minimized during perception

2. **Expected Free Energy (EFE)**
   - **Epistemic Value**: Information gain from potential actions
   - **Pragmatic Value**: Expected reward/preference satisfaction
   - **Action Selection**: Softmax over negative EFE

3. **Belief Dynamics**
   - **Prior Beliefs**: Initial state distribution
   - **Posterior Beliefs**: Updated through VFE minimization
   - **Belief Entropy**: Uncertainty measure
   - **Belief Accuracy**: Match between beliefs and true state

4. **Action-Perception Cycle**
   - **Perception**: VFE minimization updates beliefs
   - **Action**: EFE minimization selects actions
   - **Learning**: Parameter updates (if enabled)

### Implementation Matrices

1. **A Matrix** (Observation Model)
   - Maps environment states to observations
   - Represents the likelihood P(o|s)
   - Column stochastic (probabilities sum to 1)

2. **B Matrix** (Transition Model)
   - Defines state transitions for each action
   - Represents dynamics P(s'|s,a)
   - Column stochastic per action

3. **C Matrix** (Preferences)
   - Defines preferred observations
   - Encoded as log probabilities
   - Strongly prefers MEDIUM observations

4. **D Matrix** (Prior Beliefs)
   - Initial beliefs over environment states
   - Uniform prior by default

## Analysis Tools

The `biofirm_analysis.py` module provides comprehensive analysis tools:

1. **Free Energy Analysis**
   - VFE component tracking
   - EFE distributions
   - Information metrics
   - Component balance ratios

2. **Belief Dynamics Analysis**
   - Belief evolution heatmaps
   - Entropy tracking
   - Accuracy monitoring
   - Change rate analysis

3. **Action Analysis**
   - State-action distributions
   - Action entropy
   - Policy transitions
   - EFE-action relationships

4. **Visualization Suite**
   - Interactive plots
   - Time series analysis
   - State space visualization
   - Information flow diagrams

## Implementation

The implementation consists of four main components:

1. `biofirm_model.py`: Core Active Inference implementation
   - Belief updating through VFE minimization
   - Action selection using EFE
   - Matrix initialization and management

2. `biofirm_environment.py`: Environment simulation
   - True state transitions
   - Observation generation
   - Reward computation

3. `biofirm_analysis.py`: Analysis tools
   - Free energy component analysis
   - Belief dynamics visualization
   - Action-policy analysis
   - Performance metrics

4. `example.py`: Usage demonstration
   - Simulation runner
   - Visualization tools
   - Performance metrics

## Usage

To run the example simulation with analysis:

```bash
python example.py
```

This will:
1. Run a 300-step simulation
2. Generate comprehensive visualizations:
   - Free energy components
   - Belief dynamics
   - Action selection patterns
3. Create an HTML report with:
   - Summary statistics
   - Interactive plots
   - Performance metrics

## Theory

The model implements Active Inference principles:

1. **Perception** (State Estimation)
   - Minimizes VFE through belief updates
   - Balances accuracy and complexity
   - Maintains uncertainty estimates

2. **Action** (Policy Selection)
   - Minimizes EFE for action selection
   - Balances exploration and exploitation
   - Considers future outcomes

3. **Learning** (Parameter Adaptation)
   - Updates model parameters
   - Refines preferences
   - Improves predictions

## Performance Metrics

The model tracks multiple performance dimensions:

1. **Homeostatic Control**
   - Mean state deviation
   - State variance
   - Time in desired range
   - Recovery speed

2. **Information Processing**
   - Belief accuracy
   - Prediction error
   - Information gain
   - Uncertainty reduction

3. **Action Efficiency**
   - Action entropy
   - Policy consistency
   - Transition smoothness
   - Recovery patterns

## Extensions

Possible extensions to this basic model:

1. **Learning**
   - Parameter adaptation
   - Model structure learning
   - Preference learning

2. **Hierarchical Structure**
   - Multiple timescales
   - Nested control loops
   - Abstract goal states

3. **Enhanced Dynamics**
   - Continuous state spaces
   - Nonlinear transitions
   - Stochastic effects

4. **Multi-Agent Systems**
   - Coupled dynamics
   - Collective behavior
   - Emergent properties

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory?
2. Da Costa, L., et al. (2020). Active inference on discrete state-spaces.
3. Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference.
4. Buckley, C. L., et al. (2017). The free energy principle for action and perception. 
