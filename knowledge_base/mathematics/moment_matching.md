---
title: Moment Matching
description: Fitting a distribution by equating its moments to those of a target, the method of moments, its equivalence to minimising forward KL divergence in exponential families, and expectation propagation.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - approximate_inference
  - estimation
  - exponential_family
semantic_relations:
  - type: relates
    links:
      - "[[kl_divergence]]"
      - "[[exponential_family]]"
      - "[[natural_parameters]]"
      - "[[variational_inference]]"
      - "[[cumulant_expansion]]"
---

# Moment Matching

## Overview
Moment matching chooses the parameters of a simple distribution so that selected moments, usually the mean and covariance, equal those of a more complicated target. It appears in classical estimation as the method of moments and in approximate Bayesian inference as a way of summarising a complex distribution by a tractable one.

## Method of moments
Given data from a model with $k$ parameters, the method of moments equates the first $k$ population moments, expressed as functions of the parameters, with the corresponding sample moments and solves for the parameters. It is simple and consistent under mild conditions, though often less efficient than maximum likelihood.

## Moment matching and KL divergence
Let $q$ belong to an exponential family with sufficient statistics $T(x)$ and let $p$ be a target. The forward divergence $D_{\mathrm{KL}}[p\,\|\,q]$ is minimised when
$$\mathbb{E}_q[T(x)]=\mathbb{E}_p[T(x)].$$
So matching the expected sufficient statistics, which for a Gaussian are the mean and second moment, is exactly the optimal projection for that divergence. This differs from the divergence $D_{\mathrm{KL}}[q\,\|\,p]$ minimised in standard variational inference, which tends to be mode-seeking and to underestimate spread, whereas moment matching tends to cover all the mass of $p$. For example, fitting a single Gaussian to a bimodal mixture by moment matching gives a broad Gaussian between the modes.

## Expectation propagation and filtering
Expectation propagation approximates each factor of a posterior in turn. It removes the current factor from the approximation, combines it with the exact factor, and then projects the result back onto the tractable family by moment matching. Assumed density filtering applies the same projection sequentially in time, as in the update of a Gaussian approximation to a nonlinear filter. Moment matching is also used to collapse mixtures into a single component.

## Relation to Active Inference
Standard Active Inference uses variational free energy, which corresponds to the reverse divergence, together with mean-field or Laplace approximations, not moment matching. Moment-matching ideas enter in related schemes: expectation propagation is an alternative message-passing algorithm on factor graphs, and variational message passing and Bethe-type approximations are variants within the same family of methods. Active Inference does not assume that neural systems use moment matching.

## Related concepts
- [[kl_divergence]] — the divergence whose direction determines the behaviour
- [[exponential_family]] — the setting in which matching is an exact projection
- [[natural_parameters]] — the dual coordinates of the matched statistics
- [[variational_inference]] — the contrasting reverse-KL approach
- [[message_passing]] — the scheme in which expectation propagation operates
- [[gaussian_distributions]] — the usual target family

## References
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Wainwright, M. J., & Jordan, M. I. (2008). Graphical models, exponential families, and variational inference. Foundations and Trends in Machine Learning, 1, 1-305.
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
