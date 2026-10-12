---
title: Progressive Neural Networks
description: An architecture for sequential multi-task learning that adds a new network column per task, freezes earlier columns and connects them laterally, avoiding catastrophic forgetting at the cost of growing model size.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - continual_learning
  - neural_networks
  - transfer_learning
semantic_relations:
  - type: relates
    links:
      - "[[catastrophic_forgetting]]"
      - "[[stability_plasticity]]"
      - "[[continual_learning]]"
      - "[[elastic_weight_consolidation]]"
      - "[[neural_networks]]"
---

# Progressive Neural Networks

## Overview
Progressive neural networks were proposed by Rusu and colleagues (2016) as an architectural answer to the problem of learning a sequence of tasks without losing earlier skills. When a standard network is fine-tuned on a new task, the weights that encoded earlier tasks are overwritten, an effect known as catastrophic forgetting (French, 1999). Progressive networks avoid overwriting by never changing the parameters trained on earlier tasks.

## Architecture
Training on the first task produces a network, the first column, with $L$ layers. When a second task arrives, its parameters are frozen and a new column is added with fresh, trainable parameters. Layer $i$ of column $k$ receives its own previous layer's output and, through lateral connections, the outputs of the previous layer in all earlier columns:
$$h_i^{(k)}=f\Big(W_i^{(k)}h_{i-1}^{(k)}+\sum_{j<k}U_i^{(k:j)}h_{i-1}^{(j)}\Big),$$
where $W$ are the column's own weights and $U$ the lateral weights, with $f$ a nonlinearity. Features learned earlier are therefore available to the new column without being altered, which supports transfer from old tasks to new ones. Because earlier parameters are fixed, performance on those tasks cannot degrade, and the identity of the task is used to select the column at test time.

The costs are structural. The number of parameters grows with each task, and the lateral connections make the growth faster than linear in the number of tasks. The model requires a known task label at inference, and it cannot improve earlier tasks using later experience, because their columns are frozen. Alternatives address forgetting with a fixed-size network, for example regularisation that protects weights important to earlier tasks, as in elastic weight consolidation (Kirkpatrick et al., 2017), or rehearsal of stored examples. These alternatives trade the guarantee of no forgetting for a fixed memory budget.

## Relation to Active Inference
Progressive networks are a supervised and reinforcement learning method with no derivation from Active Inference. They resemble, at the level of analogy, structure-learning approaches in which a generative model is expanded with new latent components as new regularities appear while existing ones are retained. The stability-plasticity tradeoff that motivates the design is a theme shared with accounts of learning in the brain.

## Related concepts
- [[catastrophic_forgetting]] — the failure the architecture prevents
- [[stability_plasticity]] — the general dilemma behind continual learning
- [[continual_learning]] — the wider problem class
- [[elastic_weight_consolidation]] — a regularisation-based alternative
- [[neural_networks]] — the model class being extended
- [[domain_adaptation]] — transfer of learned features to new settings

## References
- Rusu, A. A., et al. (2016). Progressive neural networks. arXiv:1606.04671.
- French, R. M. (1999). Catastrophic forgetting in connectionist networks. Trends in Cognitive Sciences, 3, 128-135.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114, 3521-3526.
