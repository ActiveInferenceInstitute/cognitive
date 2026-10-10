---
title: Capacity Limits
description: The finite amount of information that working memory and attention can hold or process at once, classic estimates, slot versus resource models, and their bearing on Active Inference.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - working-memory
  - attention
  - limits
semantic_relations:
  - type: relates
    links:
      - "[[working_memory]]"
      - "[[attention_mechanisms]]"
      - "[[channel_capacity]]"
---

# Capacity Limits

## Overview
Capacity limits are the bounds on how much information can be actively maintained or concurrently processed. They are most studied in [[working_memory]] and in attention (see [[attention_limitations]]), and they constrain performance in nearly every complex task.

## Estimates and models
- **Span estimates**: Miller (1956) summarised the span of immediate memory as about seven plus or minus two items, with the unit being the chunk rather than the raw symbol. Cowan (2001) argued that when rehearsal and chunking are controlled, the underlying limit is closer to four chunks.
- **Visual working memory**: Luck and Vogel (1997) found that change-detection accuracy depended on the number of objects, not their features, with a limit of roughly three to four objects.
- **Slot versus resource models**: slot models posit a small number of discrete items stored with good precision; resource models posit a continuous pool divided among items, so that precision declines as the number of items grows. Ma, Husain and Bays (2014) review evidence favouring flexible resource accounts, with ongoing debate.
- **Chunking**: recoding items into larger meaningful units increases effective capacity without changing the item limit (see [[memory_organization]]).
- **Time-based limits**: in the phonological loop, span depends on how much can be rehearsed in about two seconds (see [[phonological_loop]]).

Limits differ across domains and individuals, correlate with measures of fluid reasoning, and are separable from the control processes that manage them (see [[central_executive]]).

## Role in Active Inference
Active Inference does not derive a specific item limit. Its link to capacity is indirect, through bounded rationality: belief updating incurs a complexity cost, the divergence of posterior from prior beliefs, which can be read as a price on information processed (see [[model_complexity]], [[variational_free_energy]]). Shannon's channel capacity provides the information-theoretic vocabulary for such bounds (see [[channel_capacity]]). Resource-style accounts also resemble precision allocation: a fixed budget of precision spread over more items lowers precision per item (see [[attention_allocation]], [[precision_weighting]]). These parallels are analogies that motivate models; they are not demonstrated explanations of the numbers measured experimentally.

## Related concepts
- [[cognitive_phenomena]] — capacity effects among broader phenomena
- [[executive_functions]] — control over limited resources
- [[information_theory]] — formal treatment of capacity

## References
- Miller, G. A. (1956). The magical number seven, plus or minus two: Some limits on our capacity for processing information. Psychological Review, 63, 81-97.
- Luck, S. J., & Vogel, E. K. (1997). The capacity of visual working memory for features and conjunctions. Nature, 390, 279-281.
- Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. Behavioral and Brain Sciences, 24, 87-114.
- Ma, W. J., Husain, M., & Bays, P. M. (2014). Changing concepts of working memory. Nature Neuroscience, 17, 347-356.
