---
title: Observational Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - learning
  - social
  - imitation
aliases:
  - social_modeling
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[social_learning]]"
      - "[[social_cognition]]"
wikidata: Q1143725
---

# Observational Learning

## Overview
Observational learning is learning from watching the behavior of others and its consequences, without direct personal practice or reinforcement. Albert Bandura's social learning theory made it central, proposing four component processes: attention to the model, retention of what was seen, reproduction of the behavior, and motivation to perform it. It covers imitation, learning which actions are rewarded or punished by watching others (vicarious reinforcement), and learning about the environment from others' choices.

## Mechanism
Several distinct mechanisms are bundled under this name:

- **Imitation**: copying the observed movement or action form.
- **Emulation**: reproducing the observed outcome or goal by one's own means.
- **Vicarious reinforcement**: updating the value of an action from another agent's outcome, which computationally resembles standard value learning with the observer's prediction error replaced by one computed on the model's outcome.
- **Inverse inference**: inferring the demonstrator's goals or beliefs from their behavior, then adopting them (as in inverse reinforcement learning).

Neurally, action observation engages premotor and parietal regions that also activate during execution; "mirror neurons" in macaque premotor cortex are the best-known single-cell evidence. The extent to which a mirror system explains imitation or understanding of others in humans is debated, and the claim that it is the mechanism of observational learning is not settled.

## Role in Active Inference
Active inference treats observing another agent as evidence about the world. If other agents are assumed to act under a model similar to one's own, then observing their actions is informative about hidden states and about which outcomes are likely or preferred, and the observer's beliefs and preferences are updated by the same inference used for first-person observations. Models of this kind appear in treatments of joint action and communication, where agents infer each other's intentions in a shared generative model. Two things should be kept clear. First, the framework gives a unified description rather than a distinct mechanism, so it does not by itself distinguish imitation from emulation. Second, the empirical mirror-system literature is largely independent of it; the connection is conceptual, via the shared idea that action observation and execution rely on common predictions.

## Related concepts
- [[social_learning]] — the broader category that includes teaching and cultural transmission
- [[social_cognition]] — inferring others' mental states, required for emulation
- [[reinforcement_learning]] — vicarious value learning builds on it
- [[associative_learning]] — conditioning can occur by observation
- [[preference_learning]] — learning another agent's preferences from behavior
- [[skill_acquisition]] — demonstration is a common route to skill

## References
- Bandura, A. (1977). *Social Learning Theory*.
- Rizzolatti, G., & Craighero, L. (2004). The mirror-neuron system. *Annual Review of Neuroscience*.
