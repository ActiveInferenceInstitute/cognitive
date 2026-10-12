---
title: Option Framework
description: The options formalism for temporal abstraction in reinforcement learning, in which closed-loop sub-policies with initiation and termination conditions extend Markov decision processes to semi-Markov decision processes.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - reinforcement_learning
  - hierarchical_control
  - temporal_abstraction
semantic_relations:
  - type: relates
    links:
      - "[[reinforcement_learning]]"
      - "[[markov_decision_process]]"
      - "[[policy_selection]]"
      - "[[hierarchical_active_inference]]"
      - "[[value_function]]"
---

# Option Framework

## Overview
The options framework, introduced by Sutton, Precup and Singh (1999), extends reinforcement learning from primitive actions to temporally extended courses of action. An option is a triple $o=(\mathcal{I}_o,\pi_o,\beta_o)$: an initiation set $\mathcal{I}_o\subseteq\mathcal{S}$ of states where the option can start, an internal policy $\pi_o(a\mid s)$, and a termination function $\beta_o:\mathcal{S}\to[0,1]$ giving the probability that the option ends in each state. Primitive actions are the special case of options that last one step.

## Formulation
Given a set of options, the agent's decisions at option boundaries form a semi-Markov decision process, in which transitions take a random number of time steps. The value of choosing an option $o$ in state $s$ satisfies a Bellman equation that sums over the random duration $k$ of the option:
$$Q(s,o)=\mathbb{E}\Big[\sum_{j=0}^{k-1}\gamma^{j}r_{t+j}+\gamma^{k}\max_{o'}Q(s_{t+k},o')\;\Big|\;s_t=s,\,o\Big]$$
for the optimal value, where the expectation is over the option's trajectory. Options can be learned with the same methods as primitive actions applied at the level of option choice. Intra-option learning updates the value estimates of all options consistent with an observed transition, so data from one option improve others.

The framework has two benefits that are usually cited. Planning over options can reach distant goals in fewer decision steps, and an option learned in one task can be reused in another. Its main open question is how useful options should be discovered, as opposed to specified by the designer. Methods include identifying bottleneck states and learning options end to end with policy-gradient methods. Poorly chosen options can also slow learning relative to primitive actions alone, because they enlarge the set of choices.

## Relation to Active Inference
Hierarchical formulations of Active Inference give higher levels slower timescales, so that a state at the upper level specifies a prior or initial condition for a sequence of lower-level transitions. This resembles an option that persists until it terminates and returns control upward. The resemblance is structural, and the two schemes differ in how they score choices, since options use value maximisation whereas Active Inference uses expected free energy with an epistemic term. The correspondence is an analogy under active study and not an established equivalence.

## Related concepts
- [[reinforcement_learning]] — the setting where options are defined
- [[markov_decision_process]] — the base model that options extend
- [[policy_selection]] — choosing among sequences of action
- [[hierarchical_active_inference]] — nested timescales with a comparable role
- [[value_function]] — values over options extend the usual definitions
- [[dynamic_programming]] — solution methods adapted to the semi-Markov case

## References
- Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence, 112, 181-211.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
