---
title: Structured Variational Inference
description: Variational approximations that retain selected dependencies in the approximate posterior, such as chain structure over time, instead of assuming full independence as in mean-field inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - variational_inference
  - approximate_inference
  - graphical_models
semantic_relations:
  - type: relates
    links:
      - "[[variational_inference]]"
      - "[[mean_field_theory]]"
      - "[[variational_message_passing]]"
      - "[[belief_updating]]"
      - "[[factor_graphs]]"
      - "[[variational_free_energy]]"
---

# Structured Variational Inference

## Overview
Variational inference approximates an intractable posterior $p(\mathbf{z}\mid\mathbf{x})$ with the closest member of a tractable family $\mathcal{Q}$, by maximising the evidence lower bound
$$\mathcal{L}(q)=\mathbb{E}_q[\log p(\mathbf{x},\mathbf{z})-\log q(\mathbf{z})]=\log p(\mathbf{x})-\mathrm{KL}\bigl(q(\mathbf{z})\,\Vert\,p(\mathbf{z}\mid\mathbf{x})\bigr).$$
Maximising $\mathcal{L}$ is equivalent to minimising the KL divergence, and it is the negative of the variational free energy. The choice of $\mathcal{Q}$ sets the trade-off between accuracy and cost.

## From mean field to structure
The mean-field family assumes full factorisation, $q(\mathbf{z})=\prod_i q_i(z_i)$. It is simple, but it discards all posterior correlations between the factors and tends to underestimate posterior variance, because the reverse KL divergence penalises placing mass where $p$ has little. A structured approximation instead keeps part of the dependence: $q(\mathbf{z})=\prod_k q_k(\mathbf{z}_{C_k})$ over larger groups $C_k$ chosen so that inference within each group stays tractable.

A standard example is the factorial hidden Markov model, whose exact posterior couples several chains through the shared observation. A structured approximation retains the chain over time within each factor and drops only the couplings between chains, so that each chain update is exact forward-backward message passing. The coordinate-ascent updates keep their mean-field form, $\log q_k^{*}\propto\mathbb{E}_{q_{-k}}[\log p(\mathbf{x},\mathbf{z})]$, but are now taken over blocks.

Related alternatives include Bethe-type approximations on factor graphs, which target marginals rather than a global factorised $q$, and expectation propagation, which minimises a different divergence.

## Relation to Active Inference
This is a direct and well-documented link. Discrete-state Active Inference commonly uses a mean-field factorisation over hidden-state factors and over time points, and the choice of approximation changes the form of the belief updates. Work on neuronal message passing has compared mean-field, Bethe and marginal approximations and discussed the different message schedules they imply. Using structured posteriors over time, for example keeping temporal dependencies between successive states, is therefore an available modelling choice and not an extension to the theory.

## Related concepts
- [[variational_inference]] — the general method of approximating a posterior by optimisation
- [[mean_field_theory]] — the fully factorised special case
- [[variational_message_passing]] — local update rules on a factor graph
- [[belief_updating]] — the inferential step in which these approximations are used
- [[variational_free_energy]] — the objective being minimised

## References
- Wainwright, M. J., & Jordan, M. I. (2008). Graphical models, exponential families, and variational inference. Foundations and Trends in Machine Learning, 1, 1-305.
- Ghahramani, Z., & Jordan, M. I. (1997). Factorial hidden Markov models. Machine Learning, 29, 245-273.
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. Journal of the American Statistical Association, 112, 859-877.
- Parr, T., Markovic, D., Kiebel, S. J., & Friston, K. J. (2019). Neuronal message passing using mean-field, Bethe, and marginal approximations. Scientific Reports, 9, 1889.
