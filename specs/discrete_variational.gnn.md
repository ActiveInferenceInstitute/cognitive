# GNN Specification: Cognitive Discrete Variational Model

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1.1

## ModelName

Cognitive Discrete Variational Model

## ModelAnnotation

Discrete two-state, two-action POMDP of the runtime in
`code/tools/src/models/active_inference/`. The canonical configuration is
`code/tools/src/models/active_inference/config_examples/discrete_variational.yaml`;
the runtime contract is `DiscreteGenerativeModel` in `generative_model.py` and
`ActiveInferenceFactory.create_from_yaml` in `dispatcher.py`.

- Two hidden states (dark, light) and two observations (dim, bright); `A` is the
  2x2 column-stochastic likelihood `A[o, s] = P(observation o | hidden state s)`.
- `B` is the transition tensor `B[s_next, s_prev, a] = P(s_next | s_prev, a)`.
  The declaration holds one 2x2 matrix per action index, matching what
  `DiscreteGenerativeModel` applies when it slices `B[:, :, action]`; action
  index 0 keeps the hidden state and index 1 flips it with probability 0.90.
- `C` holds log preferences over observations (the agent prefers "bright"), `D`
  the prior over hidden states, and `E` the habit prior over actions.
- `ModelTimeHorizon` is the planning horizon of the canonical dispatcher
  configuration (`temporal_horizon`).
- This document declares the model. The canonical YAML configuration is the
  runtime copy of the same numbers, and `code/tests/test_gnn_conformance.py`
  fails when either side is edited alone.

## StateSpaceBlock

A[2,2,type=float]        # Likelihood P(observation | hidden state); columns are states
B[2,2,2,type=float]      # Transition tensor; the parameterization holds one 2x2 matrix per action
C[2,type=float]          # Log preferences over observations
D[2,type=float]          # Prior over hidden states
E[2,type=float]          # Habit prior over actions
s[2,1,type=float]        # Hidden state distribution
s_prime[2,1,type=float]  # Next hidden state distribution
o[2,1,type=int]          # Observation (one-hot)
π[2,type=float]          # Policy distribution over actions
u[1,type=int]            # Action index
G[π,type=float]          # Expected free energy per action

## Connections

D>s
s-A
A-o
s-B
B>u
u>s_prime
C>G
E>π
G>π
π>u
s>s_prime

## InitialParameterization

# Rows are observations (dim, bright); columns are hidden states (dark, light);
# every column sums to 1 over observations.
A={
  (0.90, 0.10),
  (0.10, 0.90)
}

# One 2x2 transition matrix per action index, holding the matrices the runtime
# applies: rows are the next state, columns the previous state, and every column
# sums to 1 over next states. Action index 0 keeps the hidden state; index 1
# flips it with probability 0.90.
B={
  (
    (1.00, 0.00),
    (0.00, 1.00)
  ),
  (
    (0.10, 0.90),
    (0.90, 0.10)
  )
}

# Log preferences over observations: the agent prefers "bright".
C={(0.00, 1.00)}

# Uniform prior over the two hidden states.
D={(0.50, 0.50)}

# Uniform habit prior over the two actions.
E={(0.50, 0.50)}

## Equations

Likelihood (every column sums to 1 over observations):
  A[o, s] = P(o | s)

Transition (every column sums to 1 over next states, for each action):
  B[s_next, s_prev, a] = P(s_next | s_prev, a)

Hidden-state prediction before the next observation:
  q(s) = B[:, :, a] q(s_prev)

Expected free energy of an action, as the runtime evaluates it:
  G(a) = risk(a) - epistemic(a)

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=2

## ActInfOntologyAnnotation

A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
E=Habit
G=ExpectedFreeEnergy
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector
u=Action

## ModelParameters

num_hidden_states: 2
num_obs: 2
num_actions: 2
num_timesteps: 2
state_labels: dark, light
observation_labels: dim, bright
action_labels: keep, flip
b_tensor_order: next_state_previous_state_action
canonical_config: code/tools/src/models/active_inference/config_examples/discrete_variational.yaml

## Footer

Cognitive Discrete Variational Model — the discrete POMDP of
`code/tools/src/models/active_inference/generative_model.py`, declared here as
GNN and bound to the canonical YAML configuration by
`code/tests/test_gnn_conformance.py`. Validated with the parser revision pinned
in `specs/gnn-pin.json`.

## Signature

Hand-authored declaration; bound to the runtime model by the conformance test.
Not cryptographically signed.
