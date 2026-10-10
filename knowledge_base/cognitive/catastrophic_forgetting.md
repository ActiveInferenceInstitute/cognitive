---
title: Catastrophic Forgetting
description: The abrupt loss of previously learned knowledge when a network is trained on new data, its cause in shared weights, and mitigation by replay, regularisation and Bayesian posteriors.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - learning
  - neural-networks
  - plasticity
semantic_relations:
  - type: relates
    links:
      - "[[stability_plasticity]]"
      - "[[adaptive_resonance_theory]]"
      - "[[memory_consolidation]]"
---

# Catastrophic Forgetting

## Overview
Catastrophic forgetting, also called catastrophic interference, is the rapid and severe loss of earlier learning when a connectionist system is trained on new material without continued exposure to the old. McCloskey and Cohen (1989) demonstrated it in networks trained sequentially on arithmetic facts, and Ratcliff (1990) reported related findings; French (1999) reviewed the problem and proposed remedies. It is the sharp edge of the stability-plasticity dilemma (see [[stability_plasticity]]).

## Cause
In a distributed representation, many tasks share the same weights. Gradient descent on a new task moves the weights in directions that reduce the new loss and ignore the old, so performance on the old task can fall to chance after a modest amount of training. Sparse, non-overlapping representations reduce interference, while heavy overlap increases it.

## Mitigations
- **Interleaved training and replay**: McClelland, McNaughton and O'Reilly (1995) argued that the brain avoids the problem with complementary learning systems: the hippocampus learns quickly and then replays experience so the neocortex integrates it slowly (see [[memory_consolidation]], [[systems_consolidation]]).
- **Regularisation of important weights**: elastic weight consolidation adds a penalty $\frac{\lambda}{2}\sum_i F_i(\theta_i-\theta^{*}_{A,i})^2$ to the loss for the new task, where $F_i$ is the diagonal Fisher information for the old task and $\theta^{*}_A$ the weights after learning it (Kirkpatrick et al., 2017). This is a Gaussian approximation to the old-task posterior.
- **Architectural isolation**: allocating separate units or modules to new tasks, and resonance-based gating that creates new categories when input mismatches stored ones (see [[adaptive_resonance_theory]]).
- **Sparse coding and orthogonalisation**: reducing overlap between representations (see [[sparse_coding]]).

## Role in Active Inference
A Bayesian treatment frames the issue directly: the posterior over parameters after task A is the correct prior for task B, so exact Bayesian updating does not forget in this way (see [[bayesian_learning]]). In discrete Active Inference, Dirichlet counts accumulate across experience, so new evidence moves established beliefs less as counts grow. Forgetting in practice then arises from approximations, such as point estimates, limited capacity or non-stationary environments, and the counts framework implies the opposite risk of rigidity, since large counts slow adaptation. Elastic weight consolidation can be seen as a tractable approximation of this principle. These links are conceptual; applying them to deep, continual-learning benchmarks is still an active research area.

## Related concepts
- [[neural_plasticity]] — plasticity that makes interference possible
- [[synaptic_consolidation]] — biological mechanism that stabilises weights
- [[transfer_learning]] — the beneficial side of shared structure
- [[online_learning]] — setting where the problem arises

## References
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. Psychology of Learning and Motivation, 24, 109-165.
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. Psychological Review, 102, 419-457.
- Ratcliff, R. (1990). Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions. Psychological Review, 97, 285-308.
- French, R. M. (1999). Catastrophic forgetting in connectionist networks. Trends in Cognitive Sciences, 3, 128-135.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS, 114, 3521-3526.
