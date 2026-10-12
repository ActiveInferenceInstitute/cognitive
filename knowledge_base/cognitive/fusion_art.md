---
title: Fusion ART
description: A family of self-organizing neural networks that extends single-channel adaptive resonance theory to several input channels, so that one architecture can learn multimodal associations for unsupervised, supervised and reinforcement learning.
type: concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - neural_networks
  - adaptive_resonance
  - multimodal_learning
semantic_relations:
  - type: relates
    links:
      - "[[adaptive_resonance_theory]]"
      - "[[competitive_learning]]"
      - "[[reinforcement_learning]]"
      - "[[unsupervised_learning]]"
      - "[[episodic_memory]]"
      - "[[semantic_memory]]"
      - "[[procedural_memory]]"
---

# Fusion ART

## Overview
Fusion ART, short for fusion adaptive resonance theory, generalises the single-channel [[adaptive_resonance_theory]] (ART) models to networks with several input fields. Tan, Carpenter and Grossberg (2007) proposed it as a step towards a unified theory of learning, since machine learning has usually treated clustering, classification and control as separate problems. The same circuit can be configured for each by choosing which fields receive input.

## Architecture
A fusion ART network has $K$ input (pattern) fields and one category field. Each field $k$ receives a vector $x^k$, typically in complement-coded form, and each category node $j$ stores one weight vector $w_j^k$ per field. In the form usually given for Fuzzy ART models (Carpenter, Grossberg and Rosen, 1991), with $\wedge$ the component-wise minimum and $|\cdot|$ the sum of components:
1. **Category choice.** Each node receives the activation
$$T_j=\sum_{k=1}^{K}\gamma^k\,\frac{|x^k\wedge w_j^k|}{\alpha^k+|w_j^k|},$$
where $\gamma^k$ weights the contribution of field $k$ and $\alpha^k>0$ is a choice parameter. The node with the highest $T_j$ is selected.
2. **Template matching.** The selected node must pass a vigilance test in every field, $\dfrac{|x^k\wedge w_j^k|}{|x^k|}\ge\rho^k$. If it fails in any field, the node is reset and the search continues with the next best node. If no node passes, a new node is recruited.
3. **Learning.** In resonance, the weights move towards the match: $w_j^k\leftarrow(1-\beta^k)\,w_j^k+\beta^k\,(x^k\wedge w_j^k)$, with learning rate $\beta^k\in[0,1]$.

Because the vigilance parameters control how specific categories must be, the network can learn new categories quickly and incrementally without overwriting earlier ones, which is the stability and plasticity property of ART.

## Learning paradigms and memory
Assigning different meanings to the fields yields different tasks. With a single field the network performs unsupervised category learning. With one field for patterns and one for class labels it behaves as a supervised classifier. With fields for states, actions and rewards it supports reinforcement learning. Tan, Subagdja, Wang and Meng (2019) review the family and show how variants have been developed for unsupervised, supervised, semi-supervised, multimodal, reinforcement and sequence learning, and how they can represent episodic, semantic and procedural memory.

## Relation to Active Inference
The connection is an analogy between families of models. Both ART-type networks and Active Inference treat perception as matching a learned expectation to input, with a reset or an update when the match is poor, and both have been used to model learning and memory in agents. Fusion ART is, however, a self-organizing network with its own update rules and is not derived from the free energy principle, and the resemblance should not be read as an equivalence.

## Related concepts
- [[adaptive_resonance_theory]] — the single-channel theory that fusion ART extends
- [[competitive_learning]] — winner-take-all category selection
- [[reinforcement_learning]] — one of the paradigms the architecture supports
- [[unsupervised_learning]] — the one-field configuration
- [[episodic_memory]] — a memory type the models can represent
- [[semantic_memory]] — likewise
- [[procedural_memory]] — likewise

## References
- Carpenter, G. A., Grossberg, S., & Rosen, D. B. (1991). Fuzzy ART: Fast stable learning and categorization of analog patterns by an adaptive resonance system. Neural Networks, 4(6), 759-771. https://doi.org/10.1016/0893-6080(91)90056-B
- Tan, A.-H., Carpenter, G. A., & Grossberg, S. (2007). Intelligence through interaction: Towards a unified theory for learning. In Advances in Neural Networks - ISNN 2007 (Lecture Notes in Computer Science, pp. 1094-1103). Springer. https://doi.org/10.1007/978-3-540-72383-7_128
- Tan, A.-H., Subagdja, B., Wang, D., & Meng, L. (2019). Self-organizing neural networks for universal learning and multimodal memory encoding. Neural Networks, 120, 58-73. https://doi.org/10.1016/j.neunet.2019.08.020
