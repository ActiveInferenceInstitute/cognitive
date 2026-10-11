---
title: Natural Parameters
description: The canonical parameters of an exponential family distribution, their relation to mean parameters through the log-partition function, and their use in conjugate updates and message passing.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - exponential_family
  - statistics
  - inference
semantic_relations:
  - type: relates
    links:
      - "[[exponential_family]]"
      - "[[conjugate_priors]]"
      - "[[fisher_information]]"
      - "[[variational_message_passing]]"
      - "[[natural_gradients]]"
---

# Natural Parameters

## Overview
Natural parameters are the canonical coordinates of an exponential family. A distribution in the family can be written
$$p(x\mid\eta)=h(x)\exp\!\big(\eta^\top T(x)-A(\eta)\big),$$
where $T(x)$ are the sufficient statistics, $h(x)$ is a base measure, $\eta$ is the vector of natural parameters and $A(\eta)=\ln\int h(x)e^{\eta^\top T(x)}dx$ is the log-partition function that normalises the density. The same distribution can be described in other parametrisations, but this one makes the algebra of the family simplest.

## Examples
- **Bernoulli** with success probability $p$: $T(x)=x$ and $\eta=\ln\frac{p}{1-p}$, the log-odds or logit.
- **Categorical**: the natural parameters are log-probabilities up to an additive constant, and the probabilities are recovered by the softmax function.
- **Gaussian** with mean $\mu$ and variance $\sigma^2$: $T(x)=(x,x^2)$ and $\eta=\big(\mu/\sigma^2,\,-1/(2\sigma^2)\big)$.

## Key properties
- **Moments from derivatives**: $\nabla A(\eta)=\mathbb{E}[T(x)]$, the mean parameters, and $\nabla^2 A(\eta)=\mathrm{Cov}[T(x)]$, which is also the Fisher information matrix in these coordinates. Because $A$ is convex, natural and mean parameters are in one-to-one correspondence for a minimal family.
- **Conjugate updates**: with a conjugate prior, observing data adds the sufficient statistics of the data to the prior's natural parameters, so Bayesian updating becomes addition.
- **Product of members**: multiplying two members with the same statistics adds their natural parameters, which makes combining independent evidence straightforward.

## Relation to Active Inference
The additive structure is exploited in variational message passing for conjugate-exponential models, where each factor's update is a sum of messages expressed in natural parameters. In discrete-state Active Inference, beliefs about categorical variables are often written as softmax functions of log-evidence terms, which corresponds to working in natural parameters and then mapping back to probabilities. The Fisher-metric geometry on these coordinates also underlies natural gradient methods. This is mathematics shared with the wider variational inference literature and does not depend on the free energy principle.

## Related concepts
- [[exponential_family]] — the class of distributions parametrised this way
- [[conjugate_priors]] — priors for which updates add natural parameters
- [[fisher_information]] — the Hessian of the log-partition function
- [[variational_message_passing]] — message updates in natural parameters
- [[softmax_function]] — mapping from categorical natural parameters to probabilities
- [[gaussian_distributions]] — a family with the natural parameters shown above

## References
- Wainwright, M. J., & Jordan, M. I. (2008). Graphical models, exponential families, and variational inference. Foundations and Trends in Machine Learning, 1, 1-305.
- Winn, J., & Bishop, C. M. (2005). Variational message passing. Journal of Machine Learning Research, 6, 661-694.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
