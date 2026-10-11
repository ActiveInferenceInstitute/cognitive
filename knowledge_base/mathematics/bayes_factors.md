---
title: Bayes Factors
description: Ratios of marginal likelihoods that quantify the relative evidence two models or hypotheses receive from data, their interpretation, sensitivity to priors, and use with free energy in model comparison.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bayesian_statistics
  - model_comparison
  - evidence
semantic_relations:
  - type: relates
    links:
      - "[[model_evidence]]"
      - "[[model_comparison]]"
      - "[[hypothesis_testing]]"
      - "[[bayesian_statistics]]"
      - "[[information_criteria]]"
---

# Bayes Factors

## Overview
The Bayes factor comparing model $M_1$ with model $M_2$ on data $D$ is the ratio of their marginal likelihoods,
$$
\mathrm{BF}_{12}=\frac{p(D\mid M_1)}{p(D\mid M_2)}=\frac{\int p(D\mid\theta_1,M_1)\,p(\theta_1\mid M_1)\,d\theta_1}{\int p(D\mid\theta_2,M_2)\,p(\theta_2\mid M_2)\,d\theta_2}.
$$
By Bayes' theorem it is the factor that converts prior odds into posterior odds: $\frac{p(M_1\mid D)}{p(M_2\mid D)}=\mathrm{BF}_{12}\cdot\frac{p(M_1)}{p(M_2)}$. Values above 1 favour $M_1$, values below 1 favour $M_2$. Jeffreys (1961) proposed verbal scales for the strength of evidence, and Kass and Raftery (1995) gave a widely used variant based on $2\ln\mathrm{BF}$; such labels are conventions, not thresholds with theoretical status.

## Properties and caveats
- **Automatic penalty for complexity**: because parameters are integrated over their prior, a model that spreads probability over many outcomes is penalised relative to one that predicted the data sharply.
- **Prior dependence**: the marginal likelihood depends on the prior over parameters, and vague priors can strongly favour the simpler model (Lindley's paradox). Sensitivity analysis over reasonable priors is standard practice.
- **Nested models**: for a point null nested within a larger model, the Savage-Dickey density ratio gives the Bayes factor in favour of the point null as the ratio of posterior to prior density at the null value (under the usual condition on the prior of the nuisance parameters).
- **Computation**: the integrals are rarely analytic, so they are approximated, for instance by bridge sampling, the Laplace approximation, or variational bounds.

Unlike a p-value, a Bayes factor can express evidence for a null model as well as against it, and it is not affected by the stopping rule in the way sampling-distribution methods are.

## Relation to Active Inference
The connection is a standard working practice. Variational free energy $F$ is an upper bound on negative log model evidence, so $-F$ approximates $\ln p(D\mid M)$, and the difference in free energy between two models approximates the log Bayes factor. Bayesian model comparison and Bayesian model reduction in dynamic causal modelling use exactly this approximation, and the same quantity guides structure learning for generative models. The quality of the answer depends on how tight the bound is.

## Related concepts
- [[model_evidence]] — the marginal likelihood from which the ratio is formed
- [[model_comparison]] — the wider task Bayes factors address
- [[hypothesis_testing]] — the frequentist alternative
- [[information_criteria]] — approximate, penalty-based alternatives
- [[variational_free_energy]] — bound used to approximate log evidence

## References
- Jeffreys, H. (1961). Theory of Probability (3rd ed.). Oxford University Press.
- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. Journal of the American Statistical Association, 90, 773-795.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.
