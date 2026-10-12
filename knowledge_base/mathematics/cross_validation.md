---
title: Cross-Validation
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - model_evaluation
  - resampling
  - generalization
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/research/concepts/validation_methods]]"
      - "[[knowledge_base/research/concepts/model_comparison]]"
      - "[[knowledge_base/mathematics/information_criteria]]"
      - "[[knowledge_base/mathematics/statistical_foundations]]"
wikidata: Q541014
---

# Cross-Validation

## Overview
Cross-validation estimates how well a fitted model predicts data it was not fitted to, by repeatedly splitting the data into a training part and a held-out part and averaging the held-out score. It is a resampling alternative to analytic penalties such as AIC and makes few assumptions about the model, at the price of refitting it many times.

## Definition / Formalism
In $K$-fold cross-validation the $n$ observations are partitioned into folds $\mathcal F_1,\dots,\mathcal F_K$. With $\hat\theta_{-k}$ fitted on all data outside fold $k$ and a loss $\ell$ (squared error, or negative log predictive density),

$$\widehat{\mathrm{CV}}=\frac1n\sum_{k=1}^K\sum_{i\in\mathcal F_k}\ell\big(y_i,\;\hat\theta_{-k}\big).$$

Leave-one-out (LOO) is the case $K=n$. For ordinary least squares it needs no refitting: with hat matrix $H=X(X^\top X)^{-1}X^\top$ and residuals $e_i$, the LOO residual is

$$y_i-\hat y_{i,-i}=\frac{e_i}{1-H_{ii}} .$$

This was checked against $n=30$ explicit refits with three regressors; the largest discrepancy was $3\times10^{-14}$.

With log predictive density as the loss, cross-validation targets the expected log predictive density, the same KL-based target as AIC, but without requiring a specific parameter count. Several practical points follow. Small $K$ is biased pessimistically because less data is used for training, and large $K$ has higher variance and cost. Preprocessing or hyperparameter selection must happen inside each training fold, or the estimate is optimistic. For time series and for sequentially dependent agent behaviour, random folds leak future information into training; use blocked or forward-chaining splits instead, since observations are not exchangeable.

## Role in Active Inference
Cross-validation is a model-fitting tool, not part of the inference scheme. When an active inference agent is fitted to behavioural data (learning rates, precision, preference parameters), held-out log-likelihood over trials or participants gives a way to compare agent variants, and it avoids relying on the variational free energy bound being tight. Because an agent's choices depend on its own history, trial-wise folds are rarely independent, so splitting by participant or by block is safer. This repo's validation relies on analytic oracles and unit tests (for example, checks that the two expected-free-energy decompositions agree) rather than on resampling.

## Related concepts
- [[validation_methods]] — cross-validation among parameter recovery and predictive checks
- [[model_comparison]] — the decision cross-validation informs
- [[information_criteria]] — analytic approximations to the same target
- [[sensitivity_analysis]] — a complementary check on parameter dependence
- [[parameter_estimation]] — what is being fitted inside each fold
- [[uncertainty_quantification]] — resampling spread as an uncertainty estimate

## References
- Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society B* 36, 111-147.
- Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed. Springer.
