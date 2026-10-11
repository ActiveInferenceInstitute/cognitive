---
title: Dentate Gyrus
description: The input stage of the hippocampal formation, its granule cell circuitry and sparse coding, its proposed role in pattern separation, and adult neurogenesis in relation to generative models.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - hippocampus
  - memory
  - pattern_separation
semantic_relations:
  - type: relates
    links:
      - "[[hippocampus]]"
      - "[[pattern_separation]]"
      - "[[pattern_completion]]"
      - "[[entorhinal_cortex]]"
      - "[[neurogenesis]]"
---

# Dentate Gyrus

## Overview
The dentate gyrus is a region of the hippocampal formation that receives much of the cortical input to the hippocampus and projects to area CA3. It is a three-layered structure whose principal cells are granule cells. The molecular layer receives axons of the perforant path from layer II of the [[entorhinal_cortex]], the granule cell layer contains the cell bodies, and the polymorphic layer, or hilus, contains mossy cells and interneurons. Granule cell axons, the mossy fibres, form large terminals onto CA3 pyramidal cells.

## Coding properties
There are many more granule cells than entorhinal inputs, and granule cells fire sparsely. Marr (1971) argued that such an expansion in cell number combined with sparse activity can reduce the overlap between similar input patterns, so that CA3 can store them as distinct memories. Recordings in rats support the idea: minimal changes in the shape of the environment substantially altered correlated activity among place-modulated dentate granule cells, whereas larger changes recruited new cell populations in CA3 but not in the dentate gyrus (Leutgeb et al., 2007). In humans, high-resolution functional imaging suggests that the dentate gyrus and CA3 respond to subtle differences between similar items (Yassa and Stark, 2011). The mossy fibre synapses are considered strong enough to act as detonators for CA3 cells during encoding.

## Adult neurogenesis
New granule cells continue to be produced in the subgranular zone in adult rodents, and this is well established. Adult neurogenesis in humans has been reported but remains debated. In rodents, reducing neurogenesis can impair discrimination of similar contexts, which is consistent with a role in pattern separation, though effects depend on the task.

## Relation to Active Inference
The connection is conceptual. Discriminating similar experiences is a precondition for a generative model to maintain distinct hidden states, and the dentate gyrus is a candidate site for the sparse, decorrelated code this requires. Active Inference accounts of hippocampal function emphasise inference over spatial and episodic structure, but no established Active Inference model derives dentate gyrus properties, and the standard theory of pattern separation is formulated in terms of associative memory networks.

## Related concepts
- [[hippocampus]] — the larger formation of which the dentate gyrus is the input stage
- [[pattern_separation]] — the computation attributed to the region
- [[pattern_completion]] — the complementary process proposed for CA3
- [[entorhinal_cortex]] — the origin of the perforant path
- [[neurogenesis]] — adult production of new neurons
- [[memory_formation]] — the encoding processes it supports

## References
- Marr, D. (1971). Simple memory: a theory for archicortex. Philosophical Transactions of the Royal Society B, 262, 23-81.
- Leutgeb, J. K., Leutgeb, S., Moser, M.-B., & Moser, E. I. (2007). Pattern separation in the dentate gyrus and CA3 of the hippocampus. Science, 315, 961-966.
- Yassa, M. A., & Stark, C. E. L. (2011). Pattern separation in the hippocampus. Trends in Neurosciences, 34, 515-525.
