---
title: Variational Sequential Monte Carlo
description: A method that learns the proposal distributions of a particle filter by maximising a lower bound on the log marginal likelihood given by the sequential Monte Carlo estimator.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - variational_inference
  - monte_carlo
  - state_space_models
semantic_relations:
  - type: relates
    links:
      - "[[sequential_mc]]"
      - "[[particle_filtering]]"
      - "[[variational_inference]]"
      - "[[variational_bounds]]"
      - "[[importance_sampling]]"
      - "[[state_space_models]]"
---

# Variational Sequential Monte Carlo

## Overview
Variational sequential Monte Carlo (VSMC) combines two approximation strategies. [[sequential_mc]] gives an accurate but proposal-dependent particle approximation of the posterior in a state-space model, while [[variational_inference]] optimises a parametric family against a bound. VSMC treats the sequential Monte Carlo (SMC) algorithm itself as the variational family and tunes its proposals by gradient ascent on a bound (Naesseth et al., 2018). Closely related formulations were given as filtering variational objectives (Maddison et al., 2017) and auto-encoding SMC (Le et al., 2018).

## The objective
Let $\hat Z_N$ denote the SMC estimator of the marginal likelihood $p(y_{1:T})$ obtained with $N$ particles and proposals $q_\phi$. It is the product over time of the average unnormalised weights, and it is unbiased:
$$\mathbb{E}\bigl[\hat Z_N\bigr]=p(y_{1:T}).$$
By Jensen's inequality, $\mathbb{E}[\log\hat Z_N]\le\log p(y_{1:T})$, so the left-hand side is a valid lower bound that can be maximised over the proposal parameters $\phi$, and over model parameters when these are also learned. With a single particle ($N=1$) and no resampling it reduces to the usual evidence lower bound. As $N$ grows the bound tightens, in the same manner as the importance weighted autoencoder bound, which is the one-step case of this construction.

## Practical points
- **Gradients.** The resampling step is discrete, so the full gradient has a high-variance score-function term. A common choice is to drop it and use the reparameterised gradient of the weights, which gives a biased but low-variance estimator.
- **Cost.** Each gradient step runs a full particle filter, costing $N$ times as much as a single-sample objective.
- **Output.** After training, the learned proposals are used inside the filter, giving posterior samples with the guarantees of SMC while improving efficiency.
- **Limits.** The bound can be loose when $N$ is small or when the proposal is poor, and path degeneracy still affects smoothing estimates.

## Relation to Active Inference
Standard Active Inference uses categorical or Gaussian approximate posteriors and minimises variational free energy, which is the single-sample case of this bound. VSMC offers a way to represent multimodal or non-Gaussian beliefs within the same bound-maximisation view, at higher cost. This is a methodological extension that has been explored in machine learning more than in Active Inference, and it is not part of the canonical formulations.

## Related concepts
- [[sequential_mc]] — the particle algorithm whose estimator defines the bound
- [[particle_filtering]] — online state inference with weighted samples
- [[variational_inference]] — the optimisation view of approximate inference
- [[variational_bounds]] — lower bounds on the log evidence, of which this is one
- [[importance_sampling]] — the weighting principle behind the estimator

## References
- Naesseth, C. A., Linderman, S. W., Ranganath, R., & Blei, D. M. (2018). Variational sequential Monte Carlo. Proceedings of the 21st International Conference on Artificial Intelligence and Statistics, PMLR 84, 968-977.
- Maddison, C. J., Lawson, J., Tucker, G., Heess, N., Norouzi, M., Mnih, A., Doucet, A., & Teh, Y. W. (2017). Filtering variational objectives. Advances in Neural Information Processing Systems, 30, 6573-6583.
- Le, T. A., Igl, M., Rainforth, T., Jin, T., & Wood, F. (2018). Auto-encoding sequential Monte Carlo. International Conference on Learning Representations.
