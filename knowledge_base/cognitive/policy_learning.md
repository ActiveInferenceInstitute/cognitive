---
title: Policy Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - active_inference
  - learning
  - policy
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[policy_selection]]"
      - "[[knowledge_base/mathematics/expected_free_energy]]"
---

# Policy Learning

## Overview
In Active Inference a policy is a sequence of actions, and the agent holds a posterior distribution over policies rather than a learned mapping from states to actions. Policy learning therefore means changing the quantities that enter that posterior, not training a separate policy network.

## Core claims
**The policy posterior.** In the standard process theory, q(π) = softmax(ln E − F − γG), where G is expected free energy, E is a prior over policies, F is the variational free energy of the policy given past observations, and γ is precision. The fixed-horizon version used in this repo scores each policy by a discounted sum of per-step expected free energies, Σ_t discount^t · G_t, and returns softmax(−G/T) with temperature T playing the role of 1/γ.

**Three routes to a better policy distribution.**
1. *Habit learning.* E is updated by a Dirichlet count of which policies were selected, so repeated behaviour becomes a prior that makes it cheaper to repeat (Friston and colleagues, 2016). Policies are then chosen partly because they have been chosen.
2. *Model learning.* Learning A and B changes G, hence changes which policy is best without touching E.
3. *Structure learning.* Adding policies, states or hierarchical levels; not covered here.

**Habit versus deliberation.** Because E enters additively with −G in the log posterior, a sharply peaked E can dominate G and make behaviour insensitive to the current model; a flat E leaves the choice to G.

## Role in Active Inference
The mapping to this repo is partial. `DiscreteGenerativeModel.E` is a fixed prior over actions that never updates; policies are enumerated from the action set up to a `policy_limit`; and `InferenceConfig.policy_aggregation` determines how policies sharing a first action are combined: `MIN` (best continuation, optimistic) or `MARGINAL` (the Bayesian model average, preferring actions reachable by many adequate policies). The distribution the dispatcher returns is over first actions, not full policies. The package therefore implements policy inference, not policy learning; the habit-update rule above is the theory, and `learning_rate` in the config is a belief-blending parameter, not a habit learning rate.

## Related concepts
- [[policy_selection]] — how a posterior over policies becomes an action
- [[policy_evaluation]] — scoring policies by expected free energy
- [[prior_beliefs]] — E as the habit prior
- [[value_learning]] — learning values and models
- [[meta_learning]] — adapting the learning process itself
- [[policy_optimization]] — optimization-based alternatives

## References
- Friston, K. et al. (2016). Active inference and learning. *Neuroscience & Biobehavioral Reviews*.
- Friston, K. et al. (2017). Active inference: a process theory. *Neural Computation*.
