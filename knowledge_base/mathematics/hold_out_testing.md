---
title: Hold-Out Testing
description: Hold-out testing estimates how well a fitted model generalises by evaluating it on data withheld from training, with separate validation and test sets, and it is contrasted with cross-validation and with evidence-based model comparison.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - model_validation
  - statistical_learning
  - generalization
semantic_relations:
  - type: relates
    links:
      - "[[cross_validation]]"
      - "[[generalization]]"
      - "[[model_selection]]"
      - "[[validation_methods]]"
      - "[[model_evidence]]"
---

# Hold-Out Testing

## Overview
Hold-out testing, also called the hold-out method, assesses a predictive model by reserving part of the available data and not using it during fitting. The model is trained on the remaining data, and its performance on the reserved portion estimates how it will perform on new data from the same source. Fit to the training data is not a reliable guide, because a flexible model can match it closely and still predict poorly.

## Procedure
A common arrangement divides the data into three disjoint sets.
- **Training set.** Used to estimate the model parameters.
- **Validation set.** Used to choose hyperparameters, compare architectures or decide when to stop training.
- **Test set.** Used once, at the end, to report performance of the selected model.

Reusing the test set to make choices turns it into a second validation set, and the reported figure becomes optimistically biased. The estimate has sampling variance. For classification accuracy $p$ measured on $n$ independent test examples, the standard error is approximately $\sqrt{p(1-p)/n}$, so a small test set gives a wide interval.

## Pitfalls
- **Leakage.** Information from the test data reaching training, for example through normalisation fitted on the full data set, duplicated records, or related samples such as the same subject appearing in both sets, inflates the estimate. Splits should respect groups.
- **Dependence.** For time series the split should be chronological, so the model is not trained on the future of the test period.
- **Distribution shift.** A hold-out set from the same source does not measure performance under a changed distribution.

The method is simple and cheap, but it uses data inefficiently and its result depends on the particular split. [[cross_validation]] repeats the procedure over several splits to reduce this variability, with greater computation.

## Relation to Active Inference
Model comparison in Active Inference and in Bayesian cognitive modelling usually relies on the model evidence, approximated by the variational free energy, which penalises complexity in the objective itself and does not require withheld data. Held-out prediction remains a complementary check and is frequently reported for fitted behavioural models. The contrast is a methodological one, not a relation of derivation: a held-out log-likelihood estimates generalisation directly, while evidence measures fit under a prior.

## Related concepts
- [[cross_validation]] — repeated hold-out over multiple splits
- [[generalization]] — the property the test set estimates
- [[model_selection]] — the choices guided by the validation set
- [[model_evidence]] — the Bayesian alternative to held-out scoring
- [[model_complexity]] — the source of the gap between training and test error
- [[validation_methods]] — the wider family of validation procedures

## References
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.
- Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. Journal of the Royal Statistical Society: Series B, 36, 111-147.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
