---
title: Uncertainty
description: The degree of doubt about unknown quantities, its measurement by variance and entropy, the split between aleatoric and epistemic uncertainty, and its central role in belief updating and policy selection in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - probability
  - information_theory
  - beliefs
semantic_relations:
  - type: relates
    links:
      - "[[epistemic_uncertainty]]"
      - "[[entropy]]"
      - "[[uncertainty_quantification]]"
      - "[[precision_parameter]]"
      - "[[epistemic_value]]"
      - "[[expected_free_energy]]"
---

# Uncertainty

## Overview
Uncertainty is the degree of doubt about the value of a quantity, the outcome of an event or the state of the world. In a probabilistic treatment it is carried by the spread of a distribution over the possibilities. A belief concentrated on one value has little uncertainty, and a flat belief has much.

## Measures
For a distribution $p$, common measures are the variance, credible or confidence intervals, and the Shannon entropy $H[p]=-\sum_x p(x)\ln p(x)$, which is largest for the uniform distribution and zero for a point mass. For a Gaussian belief, the precision is the inverse variance and expresses confidence. The measures capture different aspects: variance depends on the scale of the variable, whereas entropy of a discrete distribution depends only on the probabilities.

## Aleatoric and epistemic uncertainty
Two sources are often distinguished. Aleatoric uncertainty reflects variability inherent in the process, such as sensor noise or a fair die, and cannot be reduced by collecting more data. Epistemic uncertainty reflects limited knowledge of parameters or model structure and shrinks as evidence accumulates. For a model with parameters $\theta$, predictive entropy decomposes as
$$H\big[\mathbb{E}_{\theta}\,p(y\mid\theta)\big]=\mathbb{E}_{\theta}\,H\big[p(y\mid\theta)\big]+I(y;\theta),$$
where the first term on the right is the expected aleatoric part and the mutual information $I(y;\theta)$ is the epistemic part. The division depends on the model, since what is treated as noise in one description can be a reducible ignorance in a richer one.

## Relation to Active Inference
Uncertainty is a primary quantity in the framework. Belief updating revises distributions over hidden states and parameters, and precision weights how strongly prediction errors change those beliefs. In expected free energy, the ambiguity term is the expected entropy of the likelihood and corresponds to the aleatoric type, while the epistemic value is the expected information gain about hidden states and parameters and corresponds to resolving epistemic uncertainty. Agents are therefore drawn to observations that reduce what they do not know, provided the cost in preferences is acceptable. The identification is exact within the formalism and its empirical status as a theory of behaviour is under test.

## Related concepts
- [[epistemic_uncertainty]] — reducible uncertainty from limited knowledge
- [[entropy]] — the information-theoretic measure of uncertainty
- [[uncertainty_quantification]] — methods for estimating and propagating uncertainty
- [[precision_parameter]] — inverse variance that weights beliefs
- [[epistemic_value]] — the expected reduction in uncertainty

## References
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
- Der Kiureghian, A., & Ditlevsen, O. (2009). Aleatory or epistemic? Does it matter? Structural Safety, 31, 105-112.
- Hüllermeier, E., & Waegeman, W. (2021). Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods. Machine Learning, 110, 457-506.
