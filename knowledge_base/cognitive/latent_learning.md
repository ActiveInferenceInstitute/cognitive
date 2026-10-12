---
title: Latent Learning
description: Learning that occurs without reinforcement and is revealed only later when a reward makes it useful, the classic maze experiments, and its significance for cognitive maps and exploration.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - learning
  - behaviorism
  - cognitive_maps
semantic_relations:
  - type: relates
    links:
      - "[[cognitive_maps]]"
      - "[[behaviorism]]"
      - "[[associative_learning]]"
      - "[[operant_conditioning]]"
      - "[[epistemic_value]]"
---

# Latent Learning

## Overview
Latent learning is learning that takes place in the absence of reinforcement and is not expressed in behaviour until a reason to perform appears. The term is associated with Edward Tolman, who used it to argue that learning and performance are distinct. Acquisition of knowledge about an environment can proceed silently, and an incentive then determines whether and how that knowledge is used.

## The maze experiments
In the best-known study, Tolman and Honzik (1930) ran three groups of rats through a complex maze. One group was rewarded with food at the goal on every trial, a second never received food, and a third was unrewarded for the first ten days and rewarded from the eleventh. The never-rewarded group improved little. The delayed-reward group showed a sudden fall in errors on the day after reward began, soon matching the continuously rewarded group. This abrupt shift indicated that the animals had learned the maze layout earlier, without reward, and that reward changed performance rather than creating the knowledge.

## Interpretation
Tolman took such findings to favour the view that animals acquire a cognitive map rather than only stimulus-response connections (see [[cognitive_maps]] and [[spatial_cognition]]). Learning theorists in the behaviourist tradition responded by proposing that reinforcement could occur through reduced drive or other mechanisms, and later reviews noted that results depend on procedure, species and maze design. Latent learning remains a standard example in discussions of the limits of strict [[behaviorism]] and of the difference between [[associative_learning]] and learning about structure.

## Relation to Active Inference
The fit is fairly close in principle. In Active Inference, agents are motivated by expected free energy, which includes an epistemic term: policies that reduce uncertainty about the environment are valued even when no extrinsic reward is available (see [[epistemic_value]] and [[exploration_exploitation]]). Exploration of an unrewarded maze therefore updates beliefs about transitions, and a later preference over outcomes can exploit them. Applying this to Tolman's rat data is an interpretation, since the experiments were not designed with such models in mind.

## Related concepts
- [[cognitive_maps]] — the representation latent learning is taken to support
- [[behaviorism]] — the framework the findings challenged
- [[operant_conditioning]] — reinforcement-based learning to which it is contrasted
- [[epistemic_value]] — information-seeking drive formalised in Active Inference
- [[exploration_exploitation]] — the trade-off between gathering information and using it

## References
- Tolman, E. C., & Honzik, C. H. (1930). Introduction and removal of reward, and maze performance in rats. University of California Publications in Psychology, 4, 257-275.
- Tolman, E. C. (1948). Cognitive maps in rats and men. Psychological Review, 55, 189-208.
