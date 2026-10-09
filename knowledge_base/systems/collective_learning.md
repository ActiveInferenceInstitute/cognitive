---
title: Collective Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - collective_learning
  - social_learning
  - multi_agent
  - collective_behavior
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/social_learning]]"
      - "[[knowledge_base/cognitive/multi_agent_active_inference]]"
      - "[[knowledge_base/cognitive/collective_behavior]]"
      - "[[knowledge_base/cognitive/stigmergic_coordination]]"
---

# Collective Learning

## Overview
Collective learning is the improvement of a group's performance or knowledge through the combined experience of its members, where the group-level knowledge may exceed or differ from what any single member holds. It covers both the aggregation of individual learning and the transmission of information between individuals.

## Mechanism
Several mechanisms are distinguished. In aggregation, independent estimates are combined, for example by averaging or voting; the Condorcet jury theorem shows that majority voting improves accuracy when voters are independent and individually better than chance, but correlated errors remove the gain. In social learning, individuals copy others, with strategies such as copying the majority, copying the successful, or copying when uncertain; these can speed adaptation but can also produce information cascades. In cultural accumulation, learned behavior is transmitted across generations and refined, as studied by Boyd and Richerson. In stigmergic learning, the group's memory lives in the environment: pheromone trails in ant foraging amplify successful routes and lose unsuccessful ones through evaporation, so the colony learns a route that no individual ant has surveyed.

In machine learning, parallels include multi-agent reinforcement learning, in which agents learn jointly, and federated learning (McMahan and colleagues 2017), in which devices train local models whose parameters are aggregated centrally without sharing raw data. The conditions that make collective learning succeed are diversity of information, independence or at least partial independence of errors, and a reliable aggregation rule; where social influence is strong, groups can converge on wrong beliefs.

## Role in Active Inference
Multi-agent Active Inference gives a specific account: agents that infer one another's states, or that share parts of a generative model, can coordinate and learn jointly. Kaufmann, Gupta and Taylor (2021) presented an active-inference model of collective intelligence in which agents that hold beliefs about other agents' beliefs improved collective performance in simulation. Friston and Frith (2015) discussed communication as the alignment of generative models. These are model-based demonstrations in simplified settings. In this repository the link is to [[knowledge_base/cognitive/multi_agent_active_inference]], and to the Ant_Colony Thing, where pheromone deposition makes beliefs about food location shared through the environment rather than through direct communication.

## Related concepts
- [[knowledge_base/cognitive/social_learning]] — individual learning from others
- [[knowledge_base/cognitive/multi_agent_active_inference]] — formal multi-agent treatment
- [[knowledge_base/cognitive/collective_behavior]] — group-level patterns of action
- [[knowledge_base/cognitive/stigmergic_coordination]] — environment-mediated memory
- [[knowledge_base/cognitive/reinforcement_learning]] — the single-agent learning framework that multi-agent versions extend
- [[knowledge_base/systems/distributed_cognition]] — cognition spread across people and artifacts

## References
- Kaufmann, R., Gupta, P., and Taylor, J. (2021). An active inference model of collective intelligence. Entropy.
- McMahan, H. B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. AISTATS.
- Boyd, R., and Richerson, P. J. (1985). Culture and the Evolutionary Process.
