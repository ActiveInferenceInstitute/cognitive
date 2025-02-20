---
title: Active Inference Examples in RxInfer
type: documentation
status: stable
created: 2024-03-20
tags:
  - rxinfer
  - active-inference
  - examples
semantic_relations:
  - type: implements
    links: 
      - [[active_inference]]
      - [[model_macro_paradigm]]
  - type: related
    links:
      - [[free_energy]]
      - [[message_passing]]
---

# Active Inference Examples in RxInfer

## Basic Active Inference Models

### 1. Simple Decision Making

A basic model for binary decision making with preferences:

```julia
using RxInfer, Distributions

@model function simple_decision_model(observations)
    # Prior preferences (what the agent expects/wants to observe)
    preferences ~ Normal(1.0, 0.1)  # Prefers positive outcomes
    
    # Hidden state
    state ~ Normal(0.0, 1.0)
    
    # Policy (action) selection
    π ~ Categorical([0.5, 0.5])  # Two possible actions
    
    # State transition based on policy
    next_state ~ Normal(state + π == 1 ? 0.5 : -0.5, 0.1)
    
    # Observation model
    observations ~ Normal(next_state, 0.1)
    
    return π
end

# Define constraints for variational inference
@constraints function decision_constraints()
    q(preferences, state, π, next_state) = q(preferences)q(state)q(π)q(next_state)
    
    q(preferences) :: NormalMeanPrecision
    q(state) :: NormalMeanPrecision
    q(π) :: Categorical
    q(next_state) :: NormalMeanPrecision
end
```

### 2. Multi-Step Planning

A model for planning multiple steps ahead:

```julia
@model function planning_model(T)
    # Initial state
    x = Vector{Random}(undef, T+1)
    x[1] ~ Normal(0.0, 1.0)
    
    # Policy sequence
    π = Vector{Random}(undef, T)
    for t in 1:T
        π[t] ~ Categorical([0.3, 0.4, 0.3])  # Three possible actions
    end
    
    # State transitions and observations
    for t in 1:T
        # State transition based on policy
        x[t+1] ~ Normal(
            x[t] + (π[t] == 1 ? -1.0 : π[t] == 2 ? 0.0 : 1.0),
            0.1
        )
        
        # Observations at each step
        y[t] ~ Normal(x[t], 0.1)
    end
    
    return π
end

# Define hierarchical constraints
@constraints function planning_constraints()
    # Temporal factorization
    q(x[1:T], π[1:T]) = q(x[1])∏(t->q(x[t+1])q(π[t]), 1:T)
    
    # Distribution families
    q(x[t]) :: NormalMeanPrecision
    q(π[t]) :: Categorical
end
```

## Advanced Active Inference Models

### 1. Hierarchical Perception-Action

A model with hierarchical state representation and action selection:

```julia
@model function hierarchical_active_model()
    # High-level state (context)
    context ~ Categorical([0.5, 0.5])
    
    # High-level policy
    π_high ~ Categorical([0.5, 0.5])
    
    # Context-dependent low-level state
    state ~ Normal(
        context == 1 ? 0.0 : 2.0,
        1.0
    )
    
    # Low-level policy conditioned on high-level policy
    π_low ~ Categorical(
        π_high == 1 ? [0.8, 0.2] : [0.2, 0.8]
    )
    
    # State transition
    next_state ~ Normal(
        state + (π_low == 1 ? 0.5 : -0.5),
        0.1
    )
    
    # Observations
    obs ~ Normal(next_state, 0.1)
    
    return (π_high, π_low)
end

# Define hierarchical constraints
@constraints function hierarchical_constraints()
    # Hierarchical factorization
    q(context, π_high, state, π_low, next_state) = 
        q(context, π_high)q(state)q(π_low)q(next_state)
    
    # Distribution families
    q(context) :: Categorical
    q(π_high) :: Categorical
    q(state) :: NormalMeanPrecision
    q(π_low) :: Categorical
    q(next_state) :: NormalMeanPrecision
end
```

### 2. Learning and Adaptation

A model that learns from experience:

```julia
@model function adaptive_inference_model(history)
    # Learnable parameters
    α ~ Gamma(1.0, 1.0)  # Learning rate
    β ~ Gamma(1.0, 1.0)  # Precision
    
    # State estimation with learned parameters
    state ~ Normal(0.0, 1/β)
    
    # Policy selection with adaptive exploration
    π ~ Categorical(softmax(-α * [1.0, 2.0]))
    
    # State transition with learned dynamics
    next_state ~ Normal(
        state + π == 1 ? α : -α,
        1/β
    )
    
    # Historical observations
    history .~ Normal(next_state, 1/β)
    
    return (π, α, β)
end

# Define adaptive constraints
@constraints function adaptive_constraints()
    # Parameter and state factorization
    q(α, β, state, π, next_state) = 
        q(α)q(β)q(state)q(π)q(next_state)
    
    # Distribution families
    q(α) :: GammaShapeRate
    q(β) :: GammaShapeRate
    q(state) :: NormalMeanPrecision
    q(π) :: Categorical
    q(next_state) :: NormalMeanPrecision
end
```

## Practical Usage Examples

### 1. Running Active Inference

```julia
# Create and run a simple decision model
function run_decision_model(observations)
    # Initialize model
    model = simple_decision_model()
    
    # Set up inference
    result = infer(
        model = model,
        data = (observations = observations,),
        constraints = decision_constraints()
    )
    
    # Get policy distribution
    policy_dist = result.posteriors[:π]
    
    # Select action
    action = argmax(mean(policy_dist))
    
    return action
end
```

### 2. Online Learning and Adaptation

```julia
# Online active inference with adaptation
function run_adaptive_model(env, n_steps)
    # Initialize model and history
    model = adaptive_inference_model()
    history = Float64[]
    
    for t in 1:n_steps
        # Get observation
        obs = observe(env)
        push!(history, obs)
        
        # Run inference
        result = infer(
            model = model,
            data = (history = history,),
            constraints = adaptive_constraints()
        )
        
        # Get policy and parameters
        policy = result.posteriors[:π]
        α = mean(result.posteriors[:α])
        β = mean(result.posteriors[:β])
        
        # Select and execute action
        action = sample(policy)
        execute(env, action)
    end
end
```

## Free Energy Components

### 1. Expected Free Energy Computation

```julia
# Compute expected free energy for policy evaluation
function compute_expected_free_energy(model, policy)
    # Prior preferences
    preferences = model.preferences
    
    # Predicted states under policy
    predicted_states = predict_states(model, policy)
    
    # Expected observations
    expected_obs = predict_observations(model, predicted_states)
    
    # Compute ambiguity
    ambiguity = compute_ambiguity(expected_obs)
    
    # Compute risk (deviation from preferences)
    risk = compute_risk(expected_obs, preferences)
    
    return -(ambiguity + risk)  # Negative free energy
end
```

### 2. Message Passing Implementation

```julia
# Message passing for active inference
function update_beliefs!(model, observation)
    # Forward messages (bottom-up)
    forward_messages = compute_forward_messages(
        model.nodes,
        observation
    )
    
    # Backward messages (top-down)
    backward_messages = compute_backward_messages(
        model.nodes,
        forward_messages
    )
    
    # Update beliefs
    update_node_beliefs!(
        model.nodes,
        forward_messages,
        backward_messages
    )
end
```

## References

- [[active_inference|Active Inference]]
- [[free_energy|Free Energy]]
- [[message_passing|Message Passing]]
- [[model_macro_paradigm|@model Macro]]
- [[variational_inference|Variational Inference]] 