---
title: "Expected Free Energy: Planning, Curiosity, and Goal-Directed Behavior"
type: mathematical_concept
status: active
created: 2025-01-01
updated: 2025-02-06
tags:
  - free_energy_principle
  - expected_free_energy
  - active_inference
  - planning
  - exploration_exploitation
  - epistemic_value
  - pragmatic_value
semantic_relations:
  - type: foundation
    links:
      - [[core_principle|Core Principle]]
      - [[variational_free_energy|Variational Free Energy]]
  - type: implements
    links:
      - [[knowledge_base/free_energy_principle/cognitive/decision_making|Decision Making]]
      - [[knowledge_base/free_energy_principle/AGENTS|Agent Architectures]]
  - type: relates
    links:
      - [[knowledge_base/free_energy_principle/cognitive/attention|Attention]]
      - [[knowledge_base/free_energy_principle/cognitive/learning|Learning]]
      - [[information_geometry|Information Geometry]]
---

# Expected Free Energy: Planning, Curiosity, and Goal-Directed Behavior

## Introduction

While variational free energy (VFE) governs perception (what the agent believes now), **expected free energy** (EFE) governs planning and action selection (what the agent expects to happen under different courses of action). EFE is the quantity that active inference agents minimize when selecting policies -- sequences of actions extending into the future.

The EFE is remarkable because it naturally decomposes into two components:
1. **Epistemic value** (information gain) -- driving exploration and curiosity
2. **Pragmatic value** (expected utility) -- driving exploitation and goal-directed behavior

This decomposition dissolves the exploration-exploitation dilemma: there is no separate exploration strategy needed. Exploration and exploitation emerge from minimizing a single objective.

## Definition

### The Expected Free Energy

For a policy `pi` (a sequence of actions) and a future time step `tau`, the expected free energy is:

```
G(pi) = sum_tau G(pi, tau)
```

Where for each time step:

```
G(pi, tau) = E_{q(o_tau, s_tau | pi)}[ln q(s_tau | pi) - ln p(o_tau, s_tau | pi)]
```

This is the free energy expected under the agent's predictions about future observations `o_tau` and states `s_tau` given policy `pi`.

**Crucially**: Unlike VFE, where observations `o` are given (fixed), in EFE the observations are **predicted** -- they are random variables drawn from the agent's model of the future. The agent imagines what it would observe and what states it would encounter under each policy, then evaluates the expected free energy of those imagined futures.

### Policy Selection

Policies are selected via a softmax function over negative EFE:

```
P(pi) = sigma(-gamma * G(pi)) = exp(-gamma * G(pi)) / sum_{pi'} exp(-gamma * G(pi'))
```

Where `gamma > 0` is an inverse temperature (precision) parameter:
- `gamma -> 0`: Random policy selection (maximum exploration)
- `gamma -> infinity`: Greedy policy selection (minimum EFE policy deterministically chosen)

The precision `gamma` can itself be inferred as a hidden state, implementing adaptive exploration-exploitation balance.

## Primary Decompositions

### Decomposition 1: Epistemic and Pragmatic Value

The most important decomposition of EFE:

```
G(pi, tau) = E_q(o_tau | pi)[D_KL[q(s_tau | o_tau, pi) || q(s_tau | pi)]]
              + E_q(o_tau | pi)[ln q(o_tau | pi) - ln p(o_tau)]
```

Wait -- let us derive this carefully.

Starting from:
```
G(pi, tau) = E_{q(o_tau, s_tau | pi)}[ln q(s_tau | pi) - ln p(o_tau, s_tau | pi)]
```

Factoring `p(o_tau, s_tau | pi) = p(o_tau | s_tau, pi) * p(s_tau | pi)`:

```
G = E_q[ln q(s_tau | pi) - ln p(s_tau | pi) - ln p(o_tau | s_tau, pi)]
```

Now substitute `q(s_tau | pi) = q(s_tau | o_tau, pi) * q(o_tau | pi) / q(o_tau | pi)` -- more precisely, use `ln q(s_tau | pi) = ln q(s_tau | pi)`:

The key mathematical step uses the identity:
```
E_{q(o,s)}[ln q(s|pi)] = E_{q(o,s)}[ln q(s|o,pi)] + E_{q(o,s)}[ln q(o|pi)] - E_{q(o,s)}[ln q(o|pi)]
```

Actually, let us use the direct approach. We can rewrite:
```
ln q(s_tau | pi) = ln q(s_tau | o_tau, pi) + ln q(o_tau | pi) - ln q(o_tau | s_tau, pi)
```

But this is not generally useful. The standard derivation uses a different route.

**Standard derivation**:

```
G(pi, tau) = E_{q(o_tau, s_tau | pi)}[ln q(s_tau | pi) - ln p(o_tau, s_tau | pi)]
```

Add and subtract `ln q(s_tau | o_tau, pi)`:

```
G = E_q[ln q(s_tau | pi) - ln q(s_tau | o_tau, pi)]
    + E_q[ln q(s_tau | o_tau, pi) - ln p(o_tau, s_tau | pi)]
```

The first term:
```
E_q[ln q(s_tau | pi) - ln q(s_tau | o_tau, pi)]
= -E_{q(o_tau | pi)}[D_KL[q(s_tau | o_tau, pi) || q(s_tau | pi)]]
```

This is the **negative expected information gain** -- the expected reduction in uncertainty about states after observing `o_tau`. Since we are trying to minimize G, having a large information gain (large negative contribution) makes a policy more attractive.

The second term:
```
E_q[ln q(s_tau | o_tau, pi) - ln p(o_tau, s_tau | pi)]
= E_q[ln q(s_tau | o_tau, pi) - ln p(s_tau | o_tau, pi) - ln p(o_tau | pi)]
```

If we assume `q(s_tau | o_tau, pi) approx p(s_tau | o_tau, pi)` (accurate inference):
```
approx E_q[- ln p(o_tau | pi)]
```

But `p(o_tau | pi)` vs `p(o_tau)` -- here we use the prior preferences. In the standard formulation, `p(o_tau)` encodes the agent's prior preferences (the observations it "expects" to encounter as part of its phenotype):

```
Second term approx -E_q[ln p(o_tau)]    (using prior preferences)
```

**Final decomposition**:

```
G(pi, tau) = -E_{q(o_tau|pi)}[D_KL[q(s_tau|o_tau,pi) || q(s_tau|pi)]]  -  E_{q(o_tau|pi)}[ln p(o_tau)]
              \_________________________________________/                    \____________________/
                        Epistemic Value                                        Pragmatic Value
                   (negative information gain)                              (negative expected utility)
```

**Epistemic value** (information gain / salience):
```
-E_{q(o|pi)}[D_KL[q(s|o,pi) || q(s|pi)]]
```
- Always negative or zero (reduces G)
- Large when the policy leads to observations that are highly informative about hidden states
- Drives **exploration**: policies that resolve uncertainty are preferred
- This IS curiosity -- formalized as expected information gain
- Also called **salience** or **epistemic affordance**

**Pragmatic value** (expected utility / preference satisfaction):
```
-E_{q(o|pi)}[ln p(o)]
```
- Negative when expected observations match prior preferences (low G)
- Positive when expected observations deviate from preferences (high G)
- Drives **exploitation**: policies leading to preferred observations are preferred
- `p(o)` encodes the agent's goals as a probability distribution over desired observations
- Also called **extrinsic value** or **pragmatic affordance**

### Decomposition 2: Risk and Ambiguity

An alternative decomposition reveals different aspects:

```
G(pi, tau) = E_{q(s_tau|pi)}[D_KL[q(o_tau|s_tau,pi) || p(o_tau)]]
             + E_{q(s_tau|pi)}[H[p(o_tau|s_tau)]]
```

**Risk**: `E_{q(s|pi)}[D_KL[q(o|s,pi) || p(o)]]`
- Expected KL divergence between predicted and preferred observations
- Measures expected deviation from preferred outcomes
- Drives goal-directed behavior (minimizing risk = satisfying preferences)

**Ambiguity**: `E_{q(s|pi)}[H[p(o|s)]]`
- Expected conditional entropy of observations given states
- Measures expected uncertainty in the observation mapping
- High ambiguity means the agent cannot reliably predict what it will observe even if it knows the state
- Drives **epistemic foraging**: seeking states where observations are precise and informative

**Interpretation**: Agents minimize both the risk of not achieving goals AND the ambiguity of their sensory evidence. An agent prefers policies that lead to both preferred outcomes (low risk) and reliable observations (low ambiguity).

### Decomposition 3: Instrumental and Epistemic Components (Alternative)

Yet another useful factorization:

```
G(pi, tau) = -I_{q(pi)}[o_tau; s_tau]  +  H_{q(pi)}[o_tau]  -  E_{q(o|pi)}[ln p(o)]
```

Where:
- `I[o;s]` is the mutual information between observations and states under the policy
- `H[o]` is the marginal entropy of predicted observations
- The third term is the cross-entropy between predicted and preferred observations

## The Exploration-Exploitation Balance

### Natural Resolution

Traditional approaches to exploration-exploitation (e.g., epsilon-greedy, UCB, Thompson sampling) require explicit mechanisms to balance information seeking against reward seeking. Under active inference, this balance emerges naturally:

**When uncertainty is high** (agent does not know the environment):
- Epistemic value dominates: many policies yield large information gain
- Agent explores to reduce uncertainty
- This is **curiosity-driven behavior**

**When uncertainty is low** (agent has learned the environment):
- Epistemic value diminishes: information gain becomes negligible
- Pragmatic value dominates: agent pursues preferred observations
- This is **goal-directed behavior**

**The transition is smooth and automatic** -- no switching mechanism needed. As the agent learns (uncertainty decreases), it naturally transitions from exploration to exploitation.

### Precision Modulation

The inverse temperature `gamma` modulates the overall exploration-exploitation tradeoff:

```
P(pi) = sigma(-gamma * G(pi))
```

- Higher `gamma`: More deterministic policy selection (exploit the best policy)
- Lower `gamma`: More stochastic policy selection (explore alternatives)

`gamma` can be inferred as a hidden state, creating a meta-exploration mechanism: the agent infers how confident it should be in its policy selection.

This connects to **dopaminergic modulation** in the brain, where dopamine signals encode precision over policies. See [[knowledge_base/free_energy_principle/biology/neural_systems]].

## EFE in Discrete State Spaces

### Practical Computation

For discrete state spaces with finite actions, the EFE can be computed exactly:

```
G(pi, tau) = o_tau^T * [diag(A * s_tau_pi) * (ln(A * s_tau_pi) - ln(C))]
             - diag(A^T * ln(A)) * s_tau_pi
```

Where:
- `A` is the likelihood matrix: `A_{ij} = p(o=i | s=j)`
- `s_tau_pi = B(pi) * s_{tau-1|pi}` is the predicted state at time tau under policy pi
- `C` is the prior preference vector: `C_i = ln p(o=i)`
- `o_tau = A * s_tau_pi` is the predicted observation

**Step-by-step computation**:

1. **Predict states**: `s_tau_pi = B_{a_{tau}} * s_{tau-1|pi}` (apply transition under policy action)
2. **Predict observations**: `o_tau_pi = A * s_tau_pi` (apply likelihood mapping)
3. **Compute pragmatic value**: `sum_o o_tau_pi(o) * ln(o_tau_pi(o) / C(o))`
4. **Compute epistemic value**: `sum_s s_tau_pi(s) * sum_o A(o,s) * ln A(o,s)`
5. **Sum**: `G(pi, tau) = pragmatic - epistemic`

### Example: T-Maze Task

The T-maze is the canonical example for demonstrating EFE-driven behavior:

**Setup**:
- A rat is at the center of a T-shaped maze
- A reward (food) is at one end (left or right, randomly assigned)
- A cue at the bottom of the T indicates which arm has the reward
- The rat can: go left, go right, or go to the cue location

**State factors**:
- Location: {center, left, right, cue}
- Reward location: {left, right} (hidden, does not change)

**The EFE predicts**:
1. **First move**: The rat goes to the cue location (epistemic value dominates -- it gains information about reward location)
2. **Second move**: The rat goes to the rewarded arm (pragmatic value dominates -- uncertainty is resolved, now pursue reward)

This matches empirical observations of rat behavior and illustrates the natural exploration-then-exploitation sequence.

## EFE for Continuous State Spaces

### Formulation

For continuous states and observations:

```
G(pi, tau) = integral integral q(o_tau, s_tau | pi) [ln q(s_tau | pi) - ln p(o_tau, s_tau | pi)] do ds
```

Under Gaussian assumptions:

```
G(pi, tau) approx 1/2 * [tr(Pi_o * Sigma_{o|pi}) + (mu_{o|pi} - mu_C)^T * Pi_C * (mu_{o|pi} - mu_C)]
                  - 1/2 * ln|Sigma_{s|pi}| + const
```

Where:
- First term: Expected accuracy (predicted observations vs. sensory model)
- Second term: Pragmatic value (predicted observations vs. preferred observations)
- Third term: State uncertainty (lower entropy = less epistemic drive)

### Connection to KL Control

In the continuous case, EFE minimization is closely related to **KL control** (noterov, 2007):

```
min_pi D_KL[q(tau | pi) || p(tau)]
```

Where `tau` denotes trajectories and `p(tau)` encodes the desired trajectory distribution. This connection bridges active inference with optimal control theory and the linearly-solvable MDP framework.

## Relationship to Other Frameworks

### EFE vs. Reward Maximization

| Aspect | EFE Minimization | Reward Maximization |
|--------|-----------------|-------------------|
| Objective | `min_pi G(pi)` | `max_pi E[sum_t R_t]` |
| Information seeking | Intrinsic (epistemic value) | Requires explicit bonus |
| Goal specification | Prior preferences `p(o)` | Reward function `R(s,a)` |
| Special case | EFE = -E[R] when uncertainty = 0 | - |
| Temporal horizon | Finite (planning horizon) | Can be infinite |
| Discount factor | Implicit in model structure | Explicit gamma |

### EFE vs. Information-Theoretic Exploration

EFE subsumes several information-theoretic exploration bonuses:

- **Intrinsic motivation** (Schmidhuber): Learning progress ~ rate of change of epistemic value
- **Empowerment** (Klyubin et al.): Channel capacity between actions and states ~ epistemic value
- **Curiosity** (Pathak et al.): Prediction error ~ special case of information gain
- **Count-based exploration**: Visitation counts ~ implicit uncertainty reduction

The advantage of EFE is that these are not ad-hoc bonuses added to a reward signal -- they emerge from a single principled objective.

### EFE vs. Bayesian Experimental Design

The epistemic component of EFE is formally equivalent to **Bayesian experimental design** (Lindley, 1956):

```
a* = argmax_a E_{p(o|a)}[D_KL[p(s|o,a) || p(s)]]
```

Choose the action (experiment) that maximizes expected information gain about hidden states. EFE extends this by adding the pragmatic component -- the agent seeks information AND pursues goals simultaneously.

## Sophisticated Inference

Standard EFE assumes the agent selects a full policy at the current time step and executes it open-loop. **Sophisticated inference** (Friston et al., 2021) extends this to closed-loop planning where the agent considers how it will update beliefs at each future step:

```
G_sophisticated(pi, tau) = E_{q(o_tau|pi)}[G(pi, tau+1, o_tau)]
```

The agent recursively imagines:
1. What it would observe at time tau
2. How it would update beliefs given those observations
3. What it would then do at time tau+1
4. And so on, to the planning horizon

This produces **tree-search** behavior -- the agent builds a decision tree of possible future trajectories and evaluates each. Sophisticated inference is more computationally expensive but captures the recursive nature of human planning.

## Temporal Depth and Habit Formation

### Temporal Depth

The planning horizon `T` determines how far into the future the agent considers:

```
G(pi) = sum_{tau=t+1}^{T} G(pi, tau)
```

Deeper temporal planning (larger T):
- More computationally expensive
- Enables more strategic behavior
- Can discover distal rewards

Shallow planning (small T):
- Computationally cheap
- Myopic behavior
- Fast but potentially suboptimal

### Habit Formation

As policies are repeatedly selected and confirmed, the agent can **amortize** policy selection:

```
E(pi) = -gamma * G(pi) + ln P_habit(pi)
```

Where `P_habit(pi)` is a learned prior over policies based on past successes. Initially, EFE dominates (deliberative control). Over time, the habit prior strengthens (automatic behavior).

This corresponds to the transition from **model-based** to **model-free** control in neuroscience -- from deliberative planning (prefrontal cortex) to habitual responding (basal ganglia).

## Open Questions

1. **Computational complexity**: How do biological agents compute EFE efficiently? The full computation scales exponentially with planning horizon and state space size.

2. **Monte Carlo approximations**: Can EFE be estimated with particle methods for continuous, high-dimensional problems? Recent work on "contrastive active inference" addresses this.

3. **Credit assignment**: How should EFE be decomposed across time steps for learning? This relates to the temporal difference learning problem in RL.

4. **Multi-agent EFE**: How should EFE be formulated when multiple agents interact? Each agent must model other agents' EFE computations (recursive modeling).

5. **Relationship to empowerment**: Is the epistemic component of EFE related to or distinct from empowerment (channel capacity between actions and future states)?

## Key References

1. Friston, K., Rigoli, F., Ognibene, D., Mathys, C., FitzGerald, T., & Pezzulo, G. (2015). Active inference and epistemic value. *Cognitive Neuroscience*, 6(4), 187-214.
2. Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. *Neural Computation*, 29(1), 1-49.
3. Da Costa, L., Parr, T., Sajid, N., Vesber, S., Ryan, V., & Friston, K. (2020). Active inference on discrete state-spaces: A synthesis. *Journal of Mathematical Psychology*, 99, 102447.
4. Friston, K., Da Costa, L., Sajid, N., Heins, C., Ueltzhoeffer, K., Pavliotis, G. A., & Parr, T. (2021). Sophisticated inference. *Neural Computation*, 33(3), 713-763.
5. Millidge, B., Tschantz, A., & Buckley, C. L. (2021). Whence the expected free energy? *Neural Computation*, 33(2), 447-482.
6. Sajid, N., Ball, P. J., Parr, T., & Friston, K. J. (2021). Active inference: demystified and compared. *Neural Computation*, 33(3), 674-712.
7. Tschantz, A., Baltieri, M., Seth, A. K., & Buckley, C. L. (2020). Scaling active inference. In *International Conference on Artificial General Intelligence* (pp. 399-409). Springer.

## See also

- [[knowledge_base/mathematics/expected_free_energy|Expected Free Energy]]
