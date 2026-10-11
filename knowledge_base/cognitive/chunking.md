---
title: Chunking
description: Grouping of items into larger meaningful units in memory and action, covering Miller's limit, expert chess memory, motor chunking, and the link to hierarchical models and temporal scales.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - memory
  - skill
  - hierarchy
semantic_relations:
  - type: relates
    links:
      - "[[working_memory]]"
      - "[[memory_organization]]"
      - "[[motor_control]]"
      - "[[hierarchical_models]]"
      - "[[skill_acquisition]]"
---

# Chunking

## Overview
Chunking is the recoding of several elements into a single higher-level unit. A chunk may be a familiar word made of letters, a phone number split into groups, or a movement sequence executed as one action. Because short-term capacity is limited in the number of units rather than the amount of information in each, chunking lets people hold and process more. It is a basic mechanism of memory organization and skill learning.

## Memory chunking
Miller (1956) argued that immediate memory span is about seven items, plus or minus two, and that recoding into chunks extends the information held. Later estimates for pure capacity, when rehearsal and chunking are controlled, are lower, often around four chunks (Cowan, 2001). The best-known applied demonstration is expert chess memory: masters recall realistic board positions far better than novices, but this advantage disappears for random arrangements, which suggests that experts store familiar piece patterns as units (Chase & Simon, 1973). Chunk formation is related to learning from repeated co-occurrence, and formal models describe the gradual building of larger units from smaller ones (Gobet et al., 2001). See [[working_memory]] and [[memory_organization]].

## Motor and action chunking
Practised sequences of movements, such as typing a common word or playing a musical phrase, tend to be produced in groups with short pauses between groups and fast transitions within them. This indicates that the sequence is stored hierarchically, with a chunk-level plan that triggers a set of lower-level movements. Basal ganglia circuits have been implicated in the formation and release of such action chunks (Graybiel, 1998), connecting chunking to habit and [[skill_acquisition]]. See also [[motor_control]].

## Relation to Active Inference
The link is structural and reasonably direct. Active Inference models are commonly hierarchical, with higher levels operating on slower timescales and providing context for faster lower levels. A chunk corresponds naturally to a state or sequence representation at a higher level that unfolds into a series of lower-level states or actions. Kiebel, Daunizeau and Friston (2008) showed how a hierarchy of dynamical systems with different time constants can represent and recognise sequences such as birdsong. This gives chunking a formal interpretation, but it is a model-based reading of a well-established phenomenon; whether human chunks are learned this way remains to be tested. See [[hierarchical_models]].

## Related concepts
- [[working_memory]] — the limited store that chunking extends
- [[memory_organization]] — the wider structure of memory systems
- [[motor_control]] — chunked sequences in movement
- [[hierarchical_models]] — nested timescales as a model of chunk structure
- [[skill_acquisition]] — the learning process that builds chunks

## References
- Miller, G. A. (1956). The magical number seven, plus or minus two: Some limits on our capacity for processing information. Psychological Review, 63, 81-97.
- Chase, W. G., & Simon, H. A. (1973). Perception in chess. Cognitive Psychology, 4, 55-81.
- Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. Behavioral and Brain Sciences, 24, 87-114.
- Kiebel, S. J., Daunizeau, J., & Friston, K. J. (2008). A hierarchy of time-scales and the brain. PLoS Computational Biology, 4, e1000209.
