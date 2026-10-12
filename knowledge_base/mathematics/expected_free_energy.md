---
title: Expected Free Energy
type: mathematical_concept
status: stable
created: 2024-03-20
tags:
- mathematics
- active-inference
- decision-theory
- information-theory
msc:
- 62F15
- 94A17
- 68T05
semantic_relations:
  - type: foundation
    links:
      - "[[free_energy_principle]]"
      - "[[information_theory]]"
      - "[[decision_theory]]"
  - type: implements
    links:
      - "[[active_inference]]"
      - "[[policy_selection]]"
      - "[[exploration_exploitation]]"
  - type: related
    links:
      - "[[variational_free_energy]]"
      - "[[kl_divergence]]"
      - "[[entropy]]"
---


# Expected Free Energy

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

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

The canonical decomposition separates risk (preference mismatch) and ambiguity (expected observation uncertainty):

```math

G(\pi,\tau)
= \underbrace{D_{\mathrm{KL}}\big(q(o_\tau\mid \pi)\,\|\,p(o_\tau)\big)}_{\text{Risk (preference mismatch)}}
+ \underbrace{\mathbb{E}_{q(s_\tau\mid \pi)}\big[ H[p(o_\tau\mid s_\tau)] \big]}_{\text{Ambiguity}}

```

Ambiguity already contains the expected information gain. With the joint
$q(s_\tau, o_\tau) = q(s_\tau\mid\pi)\,p(o_\tau\mid s_\tau)$, the mutual
information $I(s_\tau; o_\tau) = \mathbb{E}_{q(o_\tau\mid\pi)}\big[ D_{\mathrm{KL}}\big(q(s_\tau\mid o_\tau,\pi)\,\|\,q(s_\tau\mid \pi)\big)\big]$
is the epistemic value, and

```math

\text{Ambiguity} = H[q(o_\tau\mid\pi)] - I(s_\tau; o_\tau),
\qquad
G(\pi,\tau) = -\mathbb{E}_{q(o_\tau\mid\pi)}\big[\ln p(o_\tau)\big] - I(s_\tau; o_\tau).

```

The epistemic value is therefore a diagnostic of the same quantity, not a third
term to add to risk and ambiguity. The package reports it separately and does
not subtract it a second time; see `CHANGELOG.md`, "Corrected mathematics".

where:

- q(s_τ|π) is the predicted state distribution

- p(o_τ,s_τ) is the generative model

- q(o_τ|π) is the predicted observation distribution

- p(o_τ) is the prior preference over observations

### 3. Policy Selection

```math

P(\pi) = \sigma\big(-\gamma\,G(\pi)\big)

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

### 3. Goal-Directed Behavior

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

### 1. Optimality

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

### 2. Convergence

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

### 3. Information Bounds

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

1. Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference

1. Da Costa, L., et al. (2020). Active inference, stochastic control, and expected free energy

1. Tschantz, A., et al. (2020). Learning action-oriented models through active inference

1. Millidge, B., et al. (2021). Expected Free Energy formalizes conflict between exploration and exploitation

## Rigorous Mathematical Derivations

### Fundamental Theorem of Expected Free Energy

**Theorem** (EFE Decomposition): The expected free energy under policy $\pi$ can be uniquely decomposed as:

$$G(\pi) = \underbrace{\mathbb{E}_{Q(o,s|\pi)}[\ln Q(s|o,\pi) - \ln P(s,o)]}_{\text{Expected Variational Free Energy}} + \underbrace{D_{KL}[Q(o|\pi)\|P(o)]}_{\text{Relative Entropy}}$$

**Proof**: Starting from the definition of expected free energy:

$$G(\pi) = \mathbb{E}_{Q(o|\pi)}[F(o,\pi)]$$

where $F(o,\pi) = \mathbb{E}_{Q(s|o,\pi)}[\ln Q(s|o,\pi) - \ln P(s,o)]$ is the variational free energy.

Expanding the expectation:

\begin{align}

G(\pi) &= \int Q(o|\pi) \int Q(s|o,\pi) [\ln Q(s|o,\pi) - \ln P(s,o)] \, ds \, do \\

&= \int Q(o|\pi) Q(s|o,\pi) [\ln Q(s|o,\pi) - \ln P(s|o) - \ln P(o)] \, ds \, do \\

&= \mathbb{E}_{Q(o,s|\pi)}[\ln Q(s|o,\pi) - \ln P(s|o)] - \mathbb{E}_{Q(o|\pi)}[\ln P(o)] \\

&= \mathbb{E}_{Q(o,s|\pi)}[\ln Q(s|o,\pi) - \ln P(s|o)] + D_{KL}[Q(o|\pi)\|P(o)] \quad \blacksquare

\end{align}

### Advanced EFE Computational Framework

```python
import numpy as np

def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def efe_step(A, qs, log_C):
    """Risk + ambiguity for one predicted state distribution qs (A[o, s] = P(o|s))."""
    qo = A @ qs
    risk = np.sum(qo[qo > 0] * (np.log(qo[qo > 0]) - log_C[qo > 0]))   # KL[Q(o) || P(o)]
    ambiguity = qs @ np.array([entropy(A[:, s]) for s in range(A.shape[1])])
    return risk, ambiguity, qo

def efe_policy(A, B, log_C, q0, policy):
    """Sum of per-step EFE along a sequence of actions; B[a][s', s] = P(s'|s, a)."""
    qs, G = q0, 0.0
    for a in policy:
        qs = B[a] @ qs
        risk, ambiguity, qo = efe_step(A, qs, log_C)
        G += risk + ambiguity
    return G

# Illustrative two-state model
A = np.array([[0.9, 0.2], [0.1, 0.8]])
B = [np.eye(2), np.array([[0.2, 0.2], [0.8, 0.8]])]             # action 1 tends to move to state 1
log_C = np.log(np.array([0.2, 0.8]))
q0 = np.array([0.9, 0.1])

# Check: risk + ambiguity equals -(information gain) - E_Q(o)[ln P(o)]
risk, ambiguity, qo = efe_step(A, q0, log_C)
info_gain = entropy(qo) - ambiguity
print(risk + ambiguity, -info_gain - qo @ log_C)
```


### Advanced Policy Optimization

```python
import itertools

policies = list(itertools.product(range(2), repeat=2))          # all action sequences of length 2
G = np.array([efe_policy(A, B, log_C, q0, pi) for pi in policies])
gamma = 4.0                                                     # illustrative policy precision
p = np.exp(-gamma * (G - G.min()))
p /= p.sum()
for pi, g, pr in zip(policies, G, p):
    print(pi, round(g, 3), round(pr, 3))
```

## See also

- [[knowledge_base/free_energy_principle/mathematics/expected_free_energy|Expected Free Energy: Planning, Curiosity, and Goal-Directed Behavior]]
