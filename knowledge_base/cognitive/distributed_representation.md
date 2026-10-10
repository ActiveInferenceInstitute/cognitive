---
title: Distributed Representation
description: Distributed representations encode each item as a pattern of activity across many units, with each unit contributing to many items, in contrast to localist one-unit-per-item codes.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - representation
  - neural_coding
  - connectionism
semantic_relations:
  - type: relates
    links:
      - "[[neural_coding]]"
      - "[[population_coding]]"
      - "[[sparse_coding]]"
      - "[[knowledge_representation]]"
---

# Distributed Representation

## Overview
A distributed representation encodes an item as a pattern of activity over many units, with each unit participating in the representation of many items. A localist code instead assigns one unit (or one pool) to each item. The distinction is central to connectionist models of cognition and to debates about how cortex stores concepts and memories.

## Definition and properties
Let $n$ binary units code a set of items. A localist scheme can represent $n$ items. A distributed scheme can in principle distinguish up to $2^n$ patterns, so 10 units can separate 1024 states. Practical codes use far fewer, because items must remain separable under noise.

Key properties follow from overlap between patterns:
- **Similarity structure.** Items with related features share active units, so similarity is reflected in pattern overlap (for instance a dot product), and generalisation to similar inputs comes with the code.
- **Graceful degradation.** Damage to or noise in a few units perturbs a pattern rather than deleting an item.
- **Content-addressable storage.** Associative networks can complete a partial pattern, as in pattern completion.
- **Cost: interference.** When several items are active together their patterns superimpose, and unrelated patterns can be confused (crosstalk); this is why distributed codes need mechanisms for separating or binding items.

Hinton, McClelland and Rumelhart (1986) gave the standard treatment. Sparse coding and population coding are intermediate cases: many units, few active at once. Evidence for distributed coding in the brain includes population responses in sensory and motor cortex and multivoxel pattern analyses of imaging data, although some neurons are highly selective, so a purely distributed or purely localist view is too simple.

## Role in Active Inference
Active Inference does not require a particular code, but a generative model over latent states is naturally realised in neural populations that jointly encode a posterior distribution. Under approximate schemes such as the Laplace approximation, the sufficient statistics of a belief are carried by population activity, which is a distributed representation of uncertainty as well as of content. The link to distributed coding is therefore one of compatibility and implementation, not a derivation.

## Related concepts
- [[neural_coding]] — how information is carried by neural activity
- [[population_coding]] — decoding a variable from many neurons at once
- [[sparse_coding]] — distributed codes in which few units are active
- [[pattern_completion]] — retrieval of whole patterns from partial cues
- [[knowledge_representation]] — representational formats compared across models
- [[memory_organization]] — memory stores as distributed traces

## References
- Hinton, G. E., McClelland, J. L., & Rumelhart, D. E. (1986). Distributed representations. In Rumelhart & McClelland (Eds.), Parallel Distributed Processing, Vol. 1. MIT Press.
