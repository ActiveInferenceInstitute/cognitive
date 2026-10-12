---
title: Social Cognition
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - research
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/systems/social_systems]]"
      - "[[knowledge_base/cognitive/social_learning]]"
      - "[[knowledge_base/cognitive/multi_agent_active_inference]]"
      - "[[knowledge_base/cognitive/cooperation]]"
      - "[[knowledge_base/cognitive/communication]]"
      - "[[knowledge_base/cognitive/social_insect_cognition]]"
      - "[[knowledge_base/research/cultural_dynamics]]"
wikidata: Q1432778
---

# Social Cognition

## Overview

Social cognition is the set of mental processes that let an agent perceive, interpret and predict other agents. It covers recognising faces and actions, inferring intentions and beliefs (theory of mind, a term from Premack and Woodruff's 1978 study of chimpanzees), attending jointly to the same object, imitating, communicating and cooperating. Its development in children, its impairments (for instance in autism and some psychiatric conditions), and its neural basis are all active areas.

## Main questions

- **Mindreading.** Whether people infer others' mental states by simulating them, by applying a theory, or by a combination.
- **Action understanding.** How observed behaviour is mapped to goals. The discovery of mirror neurons in macaque premotor cortex prompted a large and disputed literature on whether they underlie action understanding.
- **Joint attention and shared intentionality.** Tomasello's account of how humans coordinate attention and goals, and how this supports language and culture.
- **Recursive reasoning.** Agents who model what others believe about them, formalised in game theory and in interactive POMDPs (Gmytrasiewicz and Doshi, 2005).

## Connection to Active Inference

The connection is an active research programme with a coherent theoretical structure and a modest empirical base. The core proposal is that an agent understands another by using its own generative model to infer the causes of the other's behaviour. A second agent is treated as a hidden state whose actions are predicted and whose goals are inferred. Kilner, Friston and Frith (2007) cast the mirror neuron system as predictive coding, in which observed actions are explained by inferring their goals through a hierarchical model. This is a proposal, and the empirical status of mirror neurons themselves remains debated.

For communication, Friston and Frith (2015) described two agents sharing a generative model so that their exchanged signals become mutually predictable and synchronise. The formal setting is the shared or "mutual" generative model, in which agents converge on common beliefs through reciprocal updates. The programme extends to multi-agent simulation, where agents infer one another's policies and preferences. Computational cost grows with the depth of recursion, so practical models truncate at low depth, and humans appear to do the same.

What is established is that the formalism can represent these tasks and reproduce qualitative behaviour in simulation. It is less clear that it explains data better than competing accounts such as simulation theory, inverse planning models or interactive POMDPs, and direct model comparisons are relatively few.

## In this repository

Related pages are [[knowledge_base/cognitive/social_learning]], [[knowledge_base/cognitive/multi_agent_active_inference]], [[knowledge_base/cognitive/cooperation]], [[knowledge_base/cognitive/communication]], [[knowledge_base/cognitive/social_insect_cognition]] and [[knowledge_base/research/cultural_dynamics]]. A study path is [[active_inference_social_learning_path]].

## References

- Premack, D., & Woodruff, G. (1978). Does the chimpanzee have a theory of mind? *Behavioral and Brain Sciences*, 1, 515-526.
- Kilner, J. M., Friston, K. J., & Frith, C. D. (2007). Predictive coding: an account of the mirror neuron system. *Cognitive Processing*, 8, 159-166.
- Friston, K., & Frith, C. (2015). A duet for one. *Consciousness and Cognition*, 36, 390-405.
- Gmytrasiewicz, P. J., & Doshi, P. (2005). A framework for sequential planning in multi-agent settings. *Journal of Artificial Intelligence Research*, 24, 49-79.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/systems/social_systems]]

## See also

- [[knowledge_base/cognitive/social_cognition|Social Cognition]]
- [[knowledge_base/free_energy_principle/cognitive/social_cognition|Social Cognition as Mutual Active Inference]]
