---
title: Continual Learning
description: Learning from a non-stationary stream of tasks or data without erasing earlier knowledge, covering catastrophic forgetting, the main method families, and the sequential Bayesian reading used in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - machine_learning
  - catastrophic_forgetting
  - plasticity
semantic_relations:
  - type: relates
    links:
      - "[[catastrophic_forgetting]]"
      - "[[stability_plasticity]]"
      - "[[transfer_learning]]"
      - "[[memory_consolidation]]"
      - "[[online_learning]]"
      - "[[belief_initialization]]"
---

# Continual Learning

## Overview
Continual learning, also called lifelong or incremental learning, concerns systems that learn from a sequence of tasks or a drifting data stream, rather than from a fixed dataset, and that must retain earlier competence while acquiring new skills. Its central difficulty is [[catastrophic_forgetting]]: when a neural network is trained on new data, gradient updates can overwrite the parameters that supported earlier tasks (McCloskey and Cohen, 1989). The underlying tension is the [[stability_plasticity]] dilemma, since a learner must be plastic enough to learn and stable enough to remember.

## Method families
- **Regularisation.** Penalise changes to parameters that mattered for earlier tasks. Elastic weight consolidation (Kirkpatrick et al., 2017) weights a quadratic penalty by an estimate of each parameter's importance, based on the diagonal of the Fisher information.
- **Replay.** Interleave stored or generated examples of past tasks with new data. This parallels the complementary learning systems account of hippocampal replay and [[memory_consolidation]] in the brain.
- **Architectural.** Allocate separate parameters, modules or growing capacity to different tasks, so later learning cannot overwrite earlier weights.

Scenarios are commonly distinguished as task-incremental, domain-incremental and class-incremental, with the last being hardest because the task identity is not given at test time. Evaluation reports average accuracy together with forgetting and with forward and backward transfer (see [[transfer_learning]]).

## Relation to Active Inference
Bayesian learning provides a principled form of continual learning. The posterior after one batch of data becomes the prior for the next, $p(\theta\mid D_1,D_2)\propto p(D_2\mid\theta)\,p(\theta\mid D_1)$, so nothing is forgotten if the posterior is represented exactly. Active Inference treats parameter learning in this way, for example by accumulating Dirichlet counts over likelihood and transition arrays. Elastic weight consolidation can be viewed as a Gaussian approximation to the previous posterior. In practice the approximations needed for large models reintroduce forgetting, so the Bayesian view is a framing and a design guide, not a solution. Reuse of a learned posterior as a starting point relates to [[belief_initialization]].

## Related concepts
- [[catastrophic_forgetting]] — the failure mode the field addresses
- [[stability_plasticity]] — the dilemma of retaining versus acquiring
- [[transfer_learning]] — reuse of knowledge across tasks
- [[memory_consolidation]] — biological mechanisms of retention and replay
- [[online_learning]] — learning from streaming data

## References
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. Psychology of Learning and Motivation, 24, 109-165.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114, 3521-3526.
- Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. Neural Networks, 113, 54-71.
