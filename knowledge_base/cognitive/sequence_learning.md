---
title: Sequence Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - learning
  - sequences
  - temporal
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[temporal_models]]"
      - "[[procedural_memory]]"
---

# Sequence Learning

## Overview
Sequence learning is the acquisition of ordered structure in events, stimuli, or actions: learning that B tends to follow A, that a series of items repeats, or that a movement pattern has a fixed order. It underlies language, music, routine behavior, and motor skill, and it can occur implicitly, without the learner being able to state the sequence.

## Mechanism
The standard behavioral paradigm is the serial reaction time task (Nissen and Bullemer, 1987), where responses to a repeating cue sequence become faster than to random order even when participants cannot report the pattern. Computationally, several ingredients recur:

- **Transition statistics**: learning first-order or higher-order transition probabilities, $P(s_{t+1} \mid s_t)$ or longer contexts.
- **Chunking**: grouping frequently co-occurring elements into single units, which shortens the effective sequence and frees working memory.
- **Hierarchical timescales**: higher levels represent slow, abstract sequence identity while lower levels represent fast item-by-item content.
- **Recurrent dynamics**: recurrent networks and sequence models store order in hidden-state trajectories.

In the brain, sequence learning involves the basal ganglia and cerebellum for motor sequences, the hippocampus for explicit and relational order (including replay), and premotor and prefrontal cortex for chunk and rule representations. Which structure contributes depends on whether the task is implicit or explicit and whether it is motor or perceptual.

## Role in Active Inference
In discrete-state active inference a sequence is expressed by the transition model (the $B$ matrices) and, in deeper models, by the dependency of one level's initial states on the states of a slower level above. Learning the sequence means accumulating Dirichlet counts on those transitions, a Hebbian-like update (see [[hebbian_learning]]). Hierarchical generative models with nested timescales were used to simulate birdsong recognition and reading, where a slow level sets the song or word and a fast level generates each element (Kiebel, Daunizeau and Friston, 2008; Friston et al., 2017). The account fits the empirical emphasis on chunking and hierarchy, but the specific neural implementation is hypothetical. Active inference also adds that learned sequences supply predictions: an unexpected element produces precision-weighted [[prediction_error]] that can be measured, for example as mismatch responses.

## Related concepts
- [[temporal_models]] — representations of ordered events
- [[procedural_memory]] — where learned motor sequences are retained
- [[statistical_learning]] — implicit extraction of transition regularities
- [[hierarchical_models]] — nested timescales that support long sequences
- [[working_memory]] — limited buffer that chunking relieves
- [[motor_learning]] — sequences are a core target of motor learning

## References
- Nissen, M. J., & Bullemer, P. (1987). Attentional requirements of learning: evidence from performance measures. *Cognitive Psychology*.
- Kiebel, S. J., Daunizeau, J., & Friston, K. J. (2008). A hierarchy of time-scales and the brain. *PLoS Computational Biology*.
- Friston, K., Rosch, R., Parr, T., Price, C., & Bowman, H. (2017). Deep temporal models and active inference. *Neuroscience & Biobehavioral Reviews*.
