---
title: Supervised Learning
description: Learning a mapping from inputs to outputs using labelled examples, covering empirical risk minimisation, generalisation, common losses, and how it relates to prediction error and unsupervised objectives in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - machine_learning
  - statistical_learning
  - prediction
semantic_relations:
  - type: relates
    links:
      - "[[unsupervised_learning]]"
      - "[[reinforcement_learning]]"
      - "[[neural_networks]]"
      - "[[prediction_error]]"
      - "[[neural_computation]]"
      - "[[deep_learning]]"
---

# Supervised Learning

## Overview
Supervised learning is the problem of learning a function $f:\mathcal{X}\to\mathcal{Y}$ from labelled examples $(x_i,y_i)$, assumed to be drawn from an unknown joint distribution $P(x,y)$. A teaching signal, the label $y_i$, tells the learner what output was correct for each input. When $\mathcal{Y}$ is continuous the task is regression, and when it is a finite set of categories the task is classification. It contrasts with [[unsupervised_learning]], which has no labels, and with [[reinforcement_learning]], which has only scalar rewards.

## Formulation
The usual approach is empirical risk minimisation: choose $f$ from a hypothesis class to minimise
$$\hat{R}(f)=\frac{1}{n}\sum_{i=1}^{n}\ell\big(f(x_i),y_i\big),$$
where $\ell$ is a loss. Squared error corresponds to a Gaussian noise model, and cross-entropy to a categorical likelihood, so both are negative log-likelihoods. The goal is a small expected loss on new data, not on the training set. The gap between the two is the generalisation error, and it is controlled by model capacity, regularisation, the amount of data and validation procedures such as cross-validation. The bias-variance trade-off describes how overly simple models underfit while overly flexible ones overfit.

Standard methods include linear and logistic regression, decision trees, support vector machines and [[neural_networks]]. Multilayer networks are usually trained by backpropagation of the loss gradient (Rumelhart et al., 1986), which underlies [[deep_learning]].

## Relation to Active Inference
Active Inference is not primarily a supervised framework, since an agent normally receives sensory outcomes rather than labels for the correct response. Several relations are nonetheless worth noting. Minimising [[prediction_error]] resembles regression, with the sensory data acting as the target of a prediction. Analyses have shown that predictive coding networks can approximate backpropagation under certain conditions, which relates a biologically motivated scheme to supervised training. Supervised objectives can also be used to fit parts of a generative model, but the agent's own objective remains free energy. The link is therefore partly formal and partly a matter of shared practice.

## Related concepts
- [[unsupervised_learning]] — learning structure without labels
- [[reinforcement_learning]] — learning from reward instead of labels
- [[neural_networks]] — the function approximators most often trained this way
- [[prediction_error]] — the signal driving learning in predictive-processing accounts
- [[neural_computation]] — computational accounts of learning in the brain

## References
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323, 533-536.
