---
title: Message Passing Networks
type: concept
status: stub
created: 2026-02-06
tags:
  - systems-theory
  - message-passing
  - graphical-models
semantic_relations:
  - type: relates
    links:
      - [[knowledge_base/mathematics/message_passing]]
      - [[knowledge_base/mathematics/factor_graphs]]
      - [[knowledge_base/mathematics/bethe_free_energy]]
      - [[network_theory]]
---

# Message Passing Networks

## Overview

Message passing networks are computational architectures where nodes exchange local information to achieve global inference. In the context of active inference and the Free Energy Principle, message passing on factor graphs provides the computational substrate for variational inference — enabling agents to update beliefs about hidden states through distributed, local computations.

## Key Concepts

### Belief Propagation
- Sum-product algorithm for exact inference on trees
- Loopy belief propagation for approximate inference on general graphs
- Connection to [[knowledge_base/mathematics/bethe_free_energy|Bethe free energy]] minimization

### Factor Graph Representation
- Bipartite graphs connecting variable and factor nodes
- Factorized representations of generative models
- Detailed in [[knowledge_base/mathematics/factor_graphs|factor graphs]]

### Variational Message Passing
- Messages derived from variational free energy minimization
- Natural gradient updates as message passing rules
- Links to [[knowledge_base/mathematics/natural_gradients|natural gradients]]

### Neural Message Passing
- Neural networks as message passing architectures
- Predictive coding as message passing in cortical hierarchies
- Connection to [[knowledge_base/cognitive/predictive_coding|predictive coding]]

## Applications in Active Inference

- Implementing perception through bottom-up and top-down message passing
- Policy evaluation via expected free energy messages
- Multi-agent coordination through shared message passing protocols
- Hierarchical inference through cascaded message passing layers

## Related Topics

- [[knowledge_base/mathematics/message_passing|Message Passing (Mathematics)]]
- [[knowledge_base/mathematics/factor_graphs|Factor Graphs]]
- [[knowledge_base/mathematics/bethe_free_energy|Bethe Free Energy]]
- [[network_theory|Network Theory]]
- [[knowledge_base/cognitive/predictive_coding|Predictive Coding]]

---

> [!note] Open Source and Licensing
> Repository: [ActiveInferenceInstitute/cognitive](https://github.com/ActiveInferenceInstitute/cognitive)
> - Documentation and knowledge base content: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
> - Code and examples: MIT License (see `LICENSE`)
