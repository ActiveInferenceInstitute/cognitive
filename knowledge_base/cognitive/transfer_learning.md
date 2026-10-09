---
title: Transfer Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - learning
  - machine_learning
  - generalization
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[meta_learning]]"
      - "[[generative_model]]"
---

# Transfer Learning

## Overview
Transfer learning is the use of knowledge acquired in one task or domain (the source) to improve learning or performance in another (the target). In psychology, transfer is positive when prior learning helps and negative when it interferes; in machine learning it usually names a set of techniques for reusing trained models.

## Mechanism
Common approaches in machine learning:

- **Pretraining and fine-tuning**: train on a large source dataset, then adapt some or all parameters on the target.
- **Feature extraction**: freeze a pretrained representation and train only a small head.
- **Domain adaptation**: align source and target distributions when the task is the same but the input distribution differs.
- **Multi-task and continual learning**: share parameters across tasks, with regularization or replay to limit interference.
- **Meta-learning**: learn an initialization or learning rule that adapts quickly to new tasks (see [[meta_learning]]).

Whether transfer helps depends on how much structure the tasks share. In human learning, transfer is typically strongest when tasks share underlying structure or rules and weak when only surface features overlap; "far transfer" between dissimilar domains is often small in empirical studies. Related notions are schemas, which let new material be assimilated into existing knowledge (see [[schema_integration]]), and analogical reasoning.

## Role in Active Inference
In active inference, transfer corresponds to reusing parts of a [[generative_model]]. If two tasks share latent structure, such as the same state-transition dynamics or the same likelihood mapping with different preferences, then the learned parameters (Dirichlet counts) from one can serve as informative priors for the other, so fewer observations are needed. Hierarchical models make this concrete: a higher level can encode abstractions shared across contexts while lower levels specialize, so only the lower levels are relearned. [[structure_learning]] provides the complementary mechanism of deciding when an existing model is adequate and when a new one must be built. These are modeling proposals consistent with the framework rather than a developed, tested theory of transfer; there is little active inference work that compares its transfer predictions against the machine learning methods above. The framework also does not remove the central difficulty, which is knowing which source knowledge is relevant to the target and which will cause negative transfer.

## Related concepts
- [[meta_learning]] — learning to adapt across tasks
- [[hierarchical_models]] — shared abstractions across contexts
- [[schema_integration]] — assimilating new knowledge into prior structure
- [[generative_model]] — the reusable knowledge store
- [[online_learning]] — sequential adaptation after transfer
- [[stability_plasticity]] — preserving old knowledge while adapting

## References
- Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE Transactions on Knowledge and Data Engineering*.
