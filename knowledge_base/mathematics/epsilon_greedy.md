---
title: Epsilon-Greedy
description: A basic exploration rule that usually takes the highest-valued action and otherwise acts at random, with its regret behaviour and its contrast with the directed exploration of Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - exploration
  - bandits
  - action_selection
semantic_relations:
  - type: relates
    links:
      - "[[exploration_exploitation]]"
      - "[[policy_selection]]"
      - "[[softmax_function]]"
      - "[[temperature_parameter]]"
      - "[[reinforcement_learning]]"
      - "[[exploration_strategies]]"
---

# Epsilon-Greedy

## Overview
Epsilon-greedy is a simple rule for balancing exploration and exploitation. At each decision the agent takes the action with the highest current value estimate with probability $1-\varepsilon$, and otherwise chooses uniformly among all actions with probability $\varepsilon$. It is the most common baseline in [[reinforcement_learning]] and in multi-armed bandit problems because it needs only value estimates and a single parameter.

## Mechanism
The agent keeps an estimate $Q_t(a)$ of each action's expected reward, for example a sample average updated incrementally by $Q\leftarrow Q+\alpha\,(r-Q)$. The selection rule is
$$\pi(a)=\begin{cases}1-\varepsilon+\varepsilon/K & a=\arg\max_{a'}Q(a')\\ \varepsilon/K & \text{otherwise,}\end{cases}$$
for $K$ actions. The exploration is undirected: random actions are chosen without regard to how uncertain each estimate is, so already well-sampled actions are tried as often as untried ones.

## Regret
With a constant $\varepsilon$ the agent keeps taking suboptimal actions forever, so cumulative regret grows linearly with the number of steps. If $\varepsilon$ decays over time, for instance in proportion to $1/t$ with a suitable constant, regret can be made to grow only logarithmically under conditions analysed by Auer, Cesa-Bianchi and Fischer (2002). Practical implementations usually anneal $\varepsilon$ on a schedule, an idea shared with [[temperature_parameter|temperature]] annealing in softmax selection. Methods such as upper confidence bounds and Thompson sampling use uncertainty directly and typically explore more efficiently.

## Relation to Active Inference
Epsilon-greedy is best read as a point of contrast. In Active Inference, exploration is not added as random noise but arises from the epistemic term of expected free energy, so policies that are expected to resolve uncertainty are favoured in a directed way (see [[policy_selection]]). Action selection is a softmax over policy values with a precision parameter, which gives graded rather than all-or-nothing randomness, closer to Boltzmann exploration than to epsilon-greedy (see [[softmax_function]]). Epsilon-greedy therefore serves as a baseline against which these directed schemes are compared.

## Related concepts
- [[exploration_exploitation]] — the trade-off the rule addresses
- [[policy_selection]] — how Active Inference chooses among policies
- [[softmax_function]] — graded stochastic selection
- [[temperature_parameter]] — the control of randomness in softmax rules
- [[exploration_strategies]] — directed and undirected alternatives

## References
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47, 235-256.
