---
title: Information Criteria
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - model_selection
  - aic
  - bic
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/research/concepts/model_comparison]]"
      - "[[knowledge_base/mathematics/kl_divergence]]"
      - "[[knowledge_base/mathematics/variational_free_energy]]"
      - "[[knowledge_base/mathematics/information_theory]]"
---

# Information Criteria

## Overview
Information criteria score fitted models by trading goodness of fit against complexity, so that a model with more parameters is not preferred merely for fitting noise. Lower values are better, and only differences between models fitted to the same data are meaningful. The two most common, AIC and BIC, answer different questions: AIC targets out-of-sample predictive accuracy, BIC approximates the marginal likelihood.

## Definition / Formalism
With maximised likelihood $\hat L$, $k$ free parameters and $n$ observations:

$$\mathrm{AIC}=2k-2\ln\hat L,\qquad \mathrm{AICc}=\mathrm{AIC}+\frac{2k(k+1)}{n-k-1},\qquad \mathrm{BIC}=k\ln n-2\ln\hat L .$$

AIC derives from an asymptotically unbiased estimate of the expected KL divergence from the true data-generating distribution to the fitted model, up to a constant common to all candidates. The AICc correction matters when $n/k$ is small (for $k=3,\,n=20$ it adds 1.5). BIC follows from a Laplace approximation to $\ln p(\text{data}\mid M)$, so $-\tfrac12\mathrm{BIC}\approx\ln p(\text{data}\mid M)$ up to an $O(1)$ term, and $\Delta\mathrm{BIC}$ approximates $-2\ln$ of a Bayes factor.

Numerical check: for 37 successes in 100 Bernoulli trials, $\ln\hat L=-65.896$, giving AIC $=133.79$ and BIC $=136.40$. The exact marginal likelihood under a uniform prior is $\ln B(38,64)$, giving $-2\ln p=136.03$, so BIC tracks the evidence to within about 0.4 here.

Criteria assume the usual regularity conditions (a well-defined maximum likelihood estimate, a parameter count that reflects effective complexity); they are unreliable for singular or hierarchical models, where WAIC or cross-validation is usually preferred.

## Role in Active Inference
The conceptual tie is strong, the practical one is looser. Variational free energy is itself an accuracy-minus-complexity quantity: $F=\mathbb{E}_q[-\ln p(o\mid s)]+D_{\mathrm{KL}}(q(s)\,\|\,p(s))$, and $-F$ lower-bounds log evidence, which is what BIC approximates crudely. Model comparison in the active inference literature therefore typically uses free energy (as a bound on evidence) directly rather than AIC or BIC. For fitting behavioural data with an active inference agent, information criteria are a standard way to compare agent variants with different numbers of free parameters. This repo does not implement them.

## Related concepts
- [[model_comparison]] — the broader workflow information criteria serve
- [[variational_free_energy]] — accuracy-complexity bound on evidence
- [[kl_divergence]] — the loss AIC estimates
- [[cross_validation]] — a resampling alternative with fewer assumptions
- [[bayesian_models]] — marginal likelihood and Bayes factors
- [[goodness_of_fit]] — the fit term in the trade-off

## References
- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control* 19, 716-723.
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics* 6, 461-464.
- Burnham, K. P. and Anderson, D. R. (2002). *Model Selection and Multimodel Inference*. Springer.
