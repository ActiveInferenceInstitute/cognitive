---
title: Elastic Weight Consolidation
description: Elastic weight consolidation limits catastrophic forgetting in neural networks by penalising changes to parameters that were important for earlier tasks, with importance given by the Fisher information.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - continual_learning
  - neural_networks
  - bayesian_learning
semantic_relations:
  - type: relates
    links:
      - "[[catastrophic_forgetting]]"
      - "[[continual_learning]]"
      - "[[fisher_information]]"
      - "[[laplace_approximation]]"
      - "[[stability_plasticity]]"
---

# Elastic Weight Consolidation

## Overview
Elastic weight consolidation (EWC) is a method for continual learning proposed by Kirkpatrick and colleagues. A network trained on task A and then on task B tends to overwrite the parameters that task A relied on, a failure known as [[catastrophic_forgetting]]. EWC adds a quadratic penalty that anchors each parameter to its value after task A in proportion to how much that parameter mattered for task A. It was motivated by synaptic consolidation, in which synapses important for a skill become less plastic.

## Formulation
Let $\theta^{\ast}_{A}$ be the parameters after training on task A. When learning task B the loss is
$$\mathcal{L}(\theta)=\mathcal{L}_B(\theta)+\frac{\lambda}{2}\sum_i F_i\,\big(\theta_i-\theta^{\ast}_{A,i}\big)^2,$$
where $\lambda$ sets how strongly the old task is protected and $F_i$ is the $i$-th diagonal element of the Fisher information matrix evaluated on task A data at $\theta^{\ast}_{A}$.

The penalty has a Bayesian justification. By Bayes' theorem, $\log p(\theta\mid D_A,D_B)=\log p(D_B\mid\theta)+\log p(\theta\mid D_A)-\log p(D_B)$, so the posterior after task A acts as the prior for task B. EWC approximates $p(\theta\mid D_A)$ by a Gaussian centred at $\theta^{\ast}_{A}$ with a diagonal precision given by $F$, which is the Laplace approximation. The Fisher information serves as a tractable stand-in for the curvature of the log posterior, and the quadratic penalty is the log of that Gaussian.

## Limitations
Using only the diagonal ignores correlations between parameters. The Gaussian approximation is local to one mode. With many tasks a separate penalty or a running combination of penalties is needed, and the stored quantities grow with the number of parameters. Alternatives include synaptic intelligence, which estimates importance along the training trajectory, and replay methods.

## Relation to Active Inference
The Bayesian reading is the strongest link. Learning in Active Inference is posterior updating over parameters, with earlier posteriors becoming priors for later experience, as when Dirichlet concentration parameters accumulate counts. EWC is an approximate way to carry such a prior through gradient-based training of a deep network. It is not part of the Active Inference formalism.

## Related concepts
- [[catastrophic_forgetting]] — the failure EWC addresses
- [[continual_learning]] — the sequential-task setting
- [[fisher_information]] — the importance measure
- [[laplace_approximation]] — the Gaussian approximation behind the penalty
- [[stability_plasticity]] — the dilemma EWC trades off
- [[synaptic_consolidation]] — the biological process that inspired it

## References
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114, 3521-3526.
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. Psychology of Learning and Motivation, 24, 109-165.
- MacKay, D. J. C. (1992). A practical Bayesian framework for backpropagation networks. Neural Computation, 4, 448-472.
