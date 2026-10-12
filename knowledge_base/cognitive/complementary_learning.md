---
title: Complementary Learning Systems
description: The theory that hippocampus and neocortex learn in complementary ways, fast episodic storage versus slow structured generalisation, which avoids catastrophic interference and relies on replay and consolidation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - memory
  - hippocampus
  - consolidation
semantic_relations:
  - type: relates
    links:
      - "[[hippocampus]]"
      - "[[memory_consolidation]]"
      - "[[systems_consolidation]]"
      - "[[stability_plasticity]]"
      - "[[catastrophic_forgetting]]"
---

# Complementary Learning Systems

## Overview
Complementary learning systems (CLS) theory proposes that the brain uses two learning systems with different properties because no single system can serve both rapid memorisation of specific episodes and gradual extraction of general structure. The [[hippocampus]] learns quickly and stores individual experiences in sparse, separated codes. The neocortex learns slowly, adjusting overlapping distributed representations that capture regularities across many experiences. The theory was set out by McClelland, McNaughton and O'Reilly (1995).

## Core argument
Networks with distributed representations that learn new items quickly suffer from catastrophic interference, in which new learning overwrites earlier knowledge (see [[catastrophic_forgetting]]). Learning in small increments, interleaved with the old material, allows the network to find a representation that accommodates both. The neocortex is therefore assumed to learn slowly, with interleaved training. To remember specific new events in the meantime, the brain needs a second system that learns in one or a few exposures without disrupting the neocortex. The hippocampus fills this role using sparse coding and [[pattern_separation]], and it can retrieve whole episodes from partial cues (see [[pattern_completion]], [[ca3]] and [[dentate_gyrus]]).

## Consolidation
In the theory, the hippocampus gradually trains the neocortex by replaying recent memories, particularly during sleep, so that cortical connections eventually support the memory independently (see [[systems_consolidation]], [[sleep]] and [[memory_consolidation]]). This offers an account of temporally graded retrograde amnesia after hippocampal damage (see [[retrograde_amnesia]]). The theory has been revised to give the cortex a role in rapid learning when new information fits an existing schema, and to allow the hippocampus a role in generalisation, and it has been extended to ideas about replay and reinforcement learning in artificial agents (Kumaran et al., 2016; O'Reilly et al., 2014). Some experimental findings, such as the persistence of detailed episodic memory in the hippocampus, are interpreted differently by alternative views.

## Relation to Active Inference
The link is an interpretive one. A generative model that captures structure across experience resembles the slowly learned neocortical component, while the hippocampus stores particular episodes that can train it. Hobson and Friston (2012) proposed that sleep plays a role in optimising generative models, by reducing redundant parameters, which is consistent with consolidation as an aspect of model selection (see [[structure_learning]] and [[generative_model]]). Formal Active Inference models explicitly implementing complementary systems are limited, so the link remains a proposal.

## Related concepts
- [[hippocampus]] — fast learner for specific episodes
- [[memory_consolidation]] — stabilisation of memories over time
- [[stability_plasticity]] — the dilemma that motivates the theory
- [[continual_learning]] — machine learning problem related to interference
- [[episodic_memory]] — memory of specific events

## References
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. Psychological Review, 102, 419-457.
- Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. Trends in Cognitive Sciences, 20, 512-534.
- O'Reilly, R. C., Bhattacharyya, R., Howard, M. D., & Ketz, N. (2014). Complementary learning systems. Cognitive Science, 38, 1229-1248.
- Hobson, J. A., & Friston, K. J. (2012). Waking and dreaming consciousness: Neurobiological and functional considerations. Progress in Neurobiology, 98, 82-98.
