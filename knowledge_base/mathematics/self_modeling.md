---
title: Self-Modeling
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - self_modeling
  - self_evidencing
  - hierarchical_models
semantic_relations:
  - type: relates
    links:
      - "[[self_awareness]]"
      - "[[metacognition]]"
      - "[[hierarchical_models]]"
      - "[[markov_blanket]]"
---

# Self-Modeling

## Overview
Self-modeling is the construction by a system of an internal model of itself: its own body, states, capabilities, and actions. In engineering it supports planning and fault recovery; in cognitive science it is proposed as a component of self-awareness.

## Definition / Formalism
In a generative-model framing, an agent's model $p(o,s,a,\theta)$ already contains terms about itself: actions $a$ (or policies $\pi$) are random variables the agent infers, $p(\pi)$ encodes beliefs about its own future behaviour, and some hidden states represent bodily or interoceptive quantities. A self-model in a stronger sense adds latent variables $z_{\text{self}}$ that explain the agent's own observation and action streams, so that

$$p(o,s,z_{\text{self}})=p(z_{\text{self}})\,p(s\mid z_{\text{self}})\,p(o\mid s,z_{\text{self}}).$$

Inference over $z_{\text{self}}$ is ordinary posterior inference; what distinguishes the model is that the referent is the agent. Sophisticated inference, which evaluates policies using beliefs about the beliefs the agent will hold in the future, is a recursive self-model of the agent's own epistemic state (Friston et al., 2021). Learned self-models in robotics (Bongard, Zykov and Lipson, 2006) estimate a robot's own morphology from motor-sensory data and use that model to plan and to recover from damage.

## Role in Active Inference
The framework supplies a vocabulary for self-models without by itself deciding what a self-model is. Hohwy's term "self-evidencing" describes an agent as gathering evidence for its own existence by minimising surprise about the states compatible with its continued existence (2016). Interoceptive inference treats emotion and bodily self as inference about internal states (Seth, 2013). The Markov blanket gives a formal boundary that defines what counts as "self" for a given description. These are theoretical proposals, and several of the stronger claims, such as that selfhood or consciousness is identified with a particular model structure, are speculative and debated. This repo's agents contain no explicit self-model beyond beliefs over their own policies and preferences.

## Related concepts
- [[self_awareness]] — cognitive treatment of self-representation
- [[metacognition]] — beliefs about one's own cognitive processes
- [[hierarchical_models]] — deeper levels can carry self-related latent causes
- [[markov_blanket]] — statistical boundary delimiting agent from world
- [[consciousness]] — wider debate about whether self-models explain experience
- [[policy_selection]] — inference over the agent's own future actions

## References
- Bongard, Zykov & Lipson (2006), Resilient machines through continuous self-modeling, Science 314.
- Hohwy (2016), The self-evidencing brain, Nous 50.
- Seth (2013), Interoceptive inference, emotion, and the embodied self, Trends in Cognitive Sciences 17.
- Friston, Da Costa, Hafner, Hesp & Parr (2021), Sophisticated inference, Neural Computation 33.
