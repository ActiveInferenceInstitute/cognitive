---
title: "Decision Making Through Expected Free Energy Minimization"
type: concept
status: active
created: 2025-01-01
updated: 2025-02-06
tags:
  - free_energy_principle
  - decision_making
  - active_inference
  - policy_selection
  - habits
  - cognitive_control
semantic_relations:
  - type: foundation
    links:
      - [[knowledge_base/free_energy_principle/mathematics/expected_free_energy|Expected Free Energy]]
      - [[knowledge_base/free_energy_principle/mathematics/core_principle|Core Principle]]
  - type: relates
    links:
      - [[perception|Perception]]
      - [[attention|Attention]]
      - [[learning|Learning]]
      - [[consciousness|Consciousness]]
  - type: implements
    links:
      - [[knowledge_base/free_energy_principle/AGENTS|Agent Architectures]]
---

# Decision Making Through Expected Free Energy Minimization

## Overview

Decision making under the Free Energy Principle is fundamentally different from classical decision theory. Rather than maximizing expected utility, active inference agents select actions by minimizing **expected free energy** (EFE). This naturally resolves the exploration-exploitation dilemma, accounts for information-seeking behavior, and provides a unified framework for habitual and goal-directed action.

## From Utility Maximization to Free Energy Minimization

### Classical Decision Theory

In expected utility theory, the agent selects the action that maximizes:

```
a* = argmax_a sum_o P(o|a) * U(o)
```

Where `U(o)` is the utility of outcome `o` and `P(o|a)` is the probability of `o` given action `a`.

**Problems with this framework**:
1. Where does `U(o)` come from? (The reward specification problem)
2. How to balance exploration vs. exploitation? (Requires separate mechanism)
3. Why do agents seek information? (No intrinsic motivation for information)
4. Why is bounded rationality rational? (Agents should maximize unboundedly)

### Active Inference Decision Theory

Under active inference, the agent selects policies `pi` (sequences of actions) by minimizing expected free energy:

```
P(pi) = sigma(-gamma * G(pi))
```

Where:
```
G(pi) = sum_tau [ -E_q(o|pi)[D_KL[q(s|o,pi) || q(s|pi)]] - E_q(o|pi)[ln p(o)] ]
         = sum_tau [ Epistemic_value(tau) + Pragmatic_value(tau) ]
```

**This resolves all four problems**:
1. Goals are encoded as prior preferences `p(o)` -- part of the generative model
2. Exploration emerges from epistemic value -- no separate mechanism
3. Information seeking IS minimizing epistemic uncertainty
4. Bounded rationality follows from finite model capacity and precision

## Policy Selection Mechanism

### The Softmax Policy Distribution

Policies are not selected deterministically but sampled from a softmax distribution:

```
P(pi) = exp(-gamma * G(pi)) / Z
```

Where `Z = sum_pi exp(-gamma * G(pi))` is the partition function and `gamma` is the inverse temperature (precision over policies).

**Low gamma** (low precision over policies):
- Flat distribution over policies
- Exploratory, stochastic behavior
- "Open-minded" -- considers many options

**High gamma** (high precision over policies):
- Peaked distribution over policies
- Exploitative, deterministic behavior
- "Committed" -- follows the best policy

### The Role of Gamma

The precision parameter `gamma` is not fixed -- it is itself inferred as a hidden state:

```
q(gamma) = Gamma(alpha, beta)  -- gamma distribution prior
```

The posterior over `gamma` depends on how well the agent's policies are performing:

- Policies consistently leading to predicted outcomes -> high gamma (more confident, more deterministic)
- Policies leading to unexpected outcomes -> low gamma (less confident, more exploratory)

**Neural correlate**: Gamma is associated with **dopaminergic** signaling. Dopamine encodes the precision of policy selection:
- High dopamine: Confident action selection (approach behavior)
- Low dopamine: Uncertain action selection (avoidance, withdrawal)

This connects to psychiatric conditions:
- **Parkinson's disease** (low dopamine): Difficulty initiating actions (low gamma)
- **Mania** (high dopamine): Impulsive, overconfident actions (high gamma)
- **Addiction**: Excessive precision over reward-seeking policies

## Deliberative vs. Habitual Decision Making

### The Dual-Process Account

Active inference provides a principled account of the transition from deliberative to habitual behavior:

**Deliberative** (model-based): Full EFE computation over policies
```
P(pi) = sigma(-gamma * G(pi))  -- expensive, flexible, slow
```

**Habitual** (model-free): Amortized policy prior
```
P(pi) = sigma(-gamma * G(pi) + ln E(pi))  -- E(pi) is the habit prior
```

Where `E(pi)` is learned from past policy selections:
```
E(pi) <- E(pi) + eta * [P(pi)_deliberative - E(pi)]
```

Over time, `E(pi)` captures the regularities in policy selection, allowing rapid action without full EFE computation.

### Neural Implementation

| Component | Neural Substrate | Function |
|-----------|-----------------|----------|
| EFE computation | Prefrontal cortex | Deliberative planning |
| Habit prior E(pi) | Basal ganglia (dorsal striatum) | Cached policy values |
| Policy precision gamma | Ventral tegmental area (dopamine) | Confidence in policy |
| Policy posterior P(pi) | Motor cortex / premotor | Final action selection |

The transition from deliberative to habitual:
1. **Novel task**: Prefrontal cortex dominates (full EFE computation)
2. **Practice**: Basal ganglia gradually encodes successful policies
3. **Expertise**: Basal ganglia dominates (habit prior sufficient)
4. **Unexpected change**: Prefrontal cortex reclaims control (EFE recomputed)

This matches the neuroscience of skill acquisition and explains phenomena like "choking under pressure" (deliberative system interfering with well-tuned habits).

## Hierarchical Decision Making

### Temporal Abstraction

Decisions operate at multiple temporal scales:

```
Level 3 (goals):     "Go to the store"        (minutes-hours)
Level 2 (subgoals):  "Walk to the car"         (seconds-minutes)
Level 1 (actions):   "Move left foot forward"  (milliseconds-seconds)
```

Each level has its own policy space and EFE:

```
G_3(pi_3) -- goal-level expected free energy
G_2(pi_2 | pi_3) -- subgoal EFE conditioned on selected goal
G_1(pi_1 | pi_2) -- action EFE conditioned on selected subgoal
```

Higher levels provide **empirical priors** for lower levels: the selected goal constrains the set of viable subgoals, which constrain viable actions.

### Options and Macro-Actions

Hierarchical active inference naturally implements the **options framework** from hierarchical RL:

- An **option** is a sub-policy with initiation and termination conditions
- Options are selected at higher levels and executed at lower levels
- The EFE of an option integrates over its entire duration

```
G(option) = sum_{t=init}^{term} G(a_t | option)
```

## Decision Making Under Uncertainty

### Ambiguity and Risk

Active inference distinguishes between:

**Risk**: Known probability of bad outcomes
```
Risk = E_q(o|pi)[D_KL[q(o|pi) || p(o)]]
```
The expected divergence between predicted and preferred observations.

**Ambiguity**: Uncertainty about the mapping from states to observations
```
Ambiguity = E_q(s|pi)[H[p(o|s)]]
```
The expected conditional entropy of observations given states.

Active inference agents are naturally **ambiguity-averse**: they avoid states where the observation model is uncertain, preferring states with clear, precise signals.

### Counterfactual Reasoning

Active inference supports counterfactual reasoning through the generative model:

```
"What would happen if I did X?" -> Simulate G(pi_X) under the generative model
```

The agent can evaluate policies it has never tried by simulating their consequences in the generative model. This is the computational basis of imagination and mental simulation.

## Connection to Reinforcement Learning

### Policy Gradient Equivalence

Under certain conditions, active inference recovers standard RL algorithms:

**When epistemic value is zero** (fully observed states):
```
G(pi) -> -E_q(o|pi)[ln p(o)] = -E_q(o|pi)[R(o)]
```

This is the negative expected reward under policy pi -- standard policy optimization.

**The policy gradient**:
```
nabla_theta P(pi | theta) = P(pi | theta) * (R(pi) - b) * nabla_theta ln P(pi | theta)
```

is recovered from the natural gradient of the EFE.

### Temporal Difference Learning

The value function in RL can be derived from the EFE:

```
V(s) = -min_pi G(pi | s) = max_pi [Epistemic_value(pi, s) + Pragmatic_value(pi, s)]
```

The TD error:
```
delta = r + gamma * V(s') - V(s)
```

corresponds to the change in (negative) expected free energy after observing a reward and transitioning to a new state. Dopaminergic prediction errors are EFE prediction errors.

## Key References

1. Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. *Neural Computation*, 29(1), 1-49.
2. Pezzulo, G., Rigoli, F., & Friston, K. J. (2018). Hierarchical active inference: a theory of motivated control. *Trends in Cognitive Sciences*, 22(4), 294-306.
3. Schwartenbeck, P., FitzGerald, T., Dolan, R. J., & Friston, K. (2013). Exploration, novelty, surprise, and free energy minimization. *Frontiers in Psychology*, 4, 710.
4. Da Costa, L., et al. (2020). Active inference on discrete state-spaces: A synthesis. *Journal of Mathematical Psychology*, 99, 102447.
5. Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference*. MIT Press. Chapters 7-9.
