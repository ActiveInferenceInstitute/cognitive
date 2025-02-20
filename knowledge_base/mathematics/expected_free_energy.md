---
title: Expected Free Energy
type: concept
status: stable
created: 2024-03-20
tags:
  - mathematics
  - active-inference
  - decision-theory
  - information-theory
semantic_relations:
  - type: foundation
    links: 
      - [[free_energy_principle]]
      - [[information_theory]]
      - [[decision_theory]]
  - type: implements
    links:
      - [[active_inference]]
      - [[policy_selection]]
      - [[exploration_exploitation]]
  - type: related
    links:
      - [[variational_free_energy]]
      - [[kl_divergence]]
      - [[entropy]]
---

# Expected Free Energy

## Overview

Expected Free Energy (EFE) is a fundamental quantity in active inference that guides action selection by balancing exploration (information gain) and exploitation (goal-seeking). It extends the [[free_energy_principle|Free Energy Principle]] to future states and actions.

## Mathematical Foundation

### 1. Basic Definition
```math
G(π) = \sum_τ G(π,τ)
```
where:
- G(π) is the expected free energy for policy π
- τ indexes future time points
- G(π,τ) is the expected free energy at time τ

### 2. Decomposition
```math
G(π,τ) = \underbrace{\mathbb{E}_{q(o_τ,s_τ|π)}[\log q(s_τ|π) - \log p(o_τ,s_τ)]}_{\text{risk}} + \underbrace{\mathbb{E}_{q(o_τ|π)}[\log q(o_τ|π) - \log p(o_τ)]}_{\text{ambiguity}}
```
where:
- q(s_τ|π) is the predicted state distribution
- p(o_τ,s_τ) is the generative model
- q(o_τ|π) is the predicted observation distribution
- p(o_τ) is the prior preference over observations

### 3. Policy Selection
```math
P(π) = σ(-γG(π))
```
where:
- σ is the softmax function
- γ is the precision parameter

## Implementation

### 1. Expected Free Energy Computation

```julia
struct ExpectedFreeEnergy
    # Time horizon
    T::Int
    # Precision parameter
    γ::Float64
    # Prior preferences
    p_o::Distribution
end

function compute_expected_free_energy(efe::ExpectedFreeEnergy,
                                   policy::Policy,
                                   model::GenerativeModel)
    # Initialize total EFE
    G_total = 0.0
    
    # Compute EFE for each future time step
    for τ in 1:efe.T
        # Predict states and observations
        q_s = predict_states(model, policy, τ)
        q_o = predict_observations(model, q_s)
        
        # Compute risk
        risk = compute_risk(q_s, q_o, model)
        
        # Compute ambiguity
        ambiguity = compute_ambiguity(q_o, efe.p_o)
        
        # Accumulate
        G_total += risk + ambiguity
    end
    
    return G_total
end
```

### 2. Risk Term

```julia
function compute_risk(q_s::Distribution,
                     q_o::Distribution,
                     model::GenerativeModel)
    # Compute KL divergence between predicted and preferred states
    risk = expectation(q_s, q_o) do s, o
        log_q_s = logpdf(q_s, s)
        log_p_so = logpdf(model.joint, (s, o))
        return log_q_s - log_p_so
    end
    
    return risk
end
```

### 3. Ambiguity Term

```julia
function compute_ambiguity(q_o::Distribution,
                          p_o::Distribution)
    # Compute epistemic value
    ambiguity = expectation(q_o) do o
        log_q_o = logpdf(q_o, o)
        log_p_o = logpdf(p_o, o)
        return log_q_o - log_p_o
    end
    
    return ambiguity
end
```

## Policy Selection

### 1. [[policy_evaluation|Policy Evaluation]]

```julia
function evaluate_policies(agent::ActiveInferenceAgent,
                         policies::Vector{Policy})
    # Compute EFE for each policy
    G = zeros(length(policies))
    
    for (i, π) in enumerate(policies)
        G[i] = compute_expected_free_energy(
            agent.efe,
            π,
            agent.gen_model
        )
    end
    
    return G
end
```

### 2. [[action_selection|Action Selection]]

```julia
function select_action(agent::ActiveInferenceAgent,
                      observation::Vector{Float64})
    # Update beliefs
    update_beliefs!(agent, observation)
    
    # Generate policies
    policies = generate_policies(agent)
    
    # Evaluate policies
    G = evaluate_policies(agent, policies)
    
    # Compute policy probabilities
    P = softmax(-agent.efe.γ * G)
    
    # Sample action
    π = sample_categorical(policies, P)
    
    return first_action(π)
end
```

### 3. [[exploration_exploitation|Exploration-Exploitation]]

```julia
function adaptive_exploration(agent::ActiveInferenceAgent,
                            temperature::Float64)
    # Modify precision based on uncertainty
    uncertainty = compute_uncertainty(agent)
    agent.efe.γ = 1.0 / (temperature * uncertainty)
    
    # Generate and evaluate policies
    policies = generate_policies(agent)
    G = evaluate_policies(agent, policies)
    
    # Select policy with adaptive exploration
    P = softmax(-agent.efe.γ * G)
    return sample_categorical(policies, P)
end
```

## Applications

### 1. [[decision_making|Decision Making]]

```julia
function make_decision(agent::ActiveInferenceAgent,
                      options::Vector{Action},
                      preferences::Distribution)
    # Set prior preferences
    agent.efe.p_o = preferences
    
    # Generate single-step policies
    policies = [Policy([a]) for a in options]
    
    # Evaluate expected free energy
    G = evaluate_policies(agent, policies)
    
    # Select option
    return options[argmin(G)]
end
```

### 2. [[active_sensing|Active Sensing]]

```julia
function active_sensing(agent::ActiveInferenceAgent,
                       environment::Environment)
    # Initialize information gain
    total_info_gain = 0.0
    
    for t in 1:agent.efe.T
        # Select action to maximize information gain
        action = select_information_seeking_action(agent)
        
        # Execute action
        observation = environment.step(action)
        
        # Update beliefs and compute information gain
        info_gain = update_and_compute_gain!(agent, observation)
        total_info_gain += info_gain
    end
    
    return total_info_gain
end
```

### 3. [[goal_directed_behavior|Goal-Directed Behavior]]

```julia
function goal_directed_policy(agent::ActiveInferenceAgent,
                            goal_state::State)
    # Set prior preferences to favor goal state
    set_goal_preference!(agent, goal_state)
    
    # Generate multi-step policies
    policies = generate_goal_directed_policies(agent, goal_state)
    
    # Evaluate policies considering both goal and information gain
    G = zeros(length(policies))
    for (i, π) in enumerate(policies)
        # Compute expected free energy
        G[i] = compute_expected_free_energy(agent.efe, π, agent.gen_model)
        
        # Add goal-specific term
        G[i] += compute_goal_distance(π, goal_state)
    end
    
    return policies[argmin(G)]
end
```

## Theoretical Results

### 1. [[optimality|Optimality]]

```julia
function prove_optimality(efe::ExpectedFreeEnergy)
    # Demonstrate that minimizing EFE leads to optimal behavior
    
    # 1. Information gain is maximized
    show_information_maximization(efe)
    
    # 2. Goal-seeking behavior emerges
    show_goal_directed_behavior(efe)
    
    # 3. Uncertainty is minimized
    show_uncertainty_reduction(efe)
end
```

### 2. [[convergence|Convergence]]

```julia
function analyze_convergence(agent::ActiveInferenceAgent,
                           environment::Environment)
    # Track EFE over time
    G_history = Float64[]
    
    while !converged(agent)
        # Select and execute action
        action = select_action(agent, observe(environment))
        environment.step(action)
        
        # Record EFE
        push!(G_history, compute_current_efe(agent))
        
        # Update beliefs
        update_beliefs!(agent, observe(environment))
    end
    
    return G_history
end
```

### 3. [[information_bounds|Information Bounds]]

```julia
function compute_information_bounds(efe::ExpectedFreeEnergy)
    # Compute upper bound on information gain
    max_info_gain = compute_max_information_gain(efe)
    
    # Compute lower bound on expected free energy
    min_efe = compute_minimum_efe(efe)
    
    # Compute bounds on policy entropy
    H_bounds = compute_policy_entropy_bounds(efe)
    
    return (max_info_gain, min_efe, H_bounds)
end
```

## Best Practices

### 1. Implementation
- Use numerically stable computations
- Implement efficient policy search
- Cache intermediate results
- Handle edge cases

### 2. Tuning
- Adjust precision parameter
- Balance exploration-exploitation
- Set appropriate time horizon
- Define meaningful preferences

### 3. Validation
- Test with known solutions
- Verify information gains
- Monitor convergence
- Validate actions

## References

1. Friston, K. J., et al. (2015). Active inference and epistemic value
2. Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference
3. Da Costa, L., et al. (2020). Active inference, stochastic control, and expected free energy
4. Tschantz, A., et al. (2020). Learning action-oriented models through active inference
5. Millidge, B., et al. (2021). Expected Free Energy formalizes conflict between exploration and exploitation 