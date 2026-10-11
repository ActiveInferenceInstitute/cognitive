---
title: Epistemic Uncertainty
description: Uncertainty that arises from limited knowledge and can in principle be reduced by more data, contrasted with aleatoric uncertainty, with its estimation in machine learning and role in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - uncertainty
  - bayesian_inference
  - information_seeking
semantic_relations:
  - type: relates
    links:
      - "[[uncertainty]]"
      - "[[uncertainty_quantification]]"
      - "[[epistemic_value]]"
      - "[[bayesian_deep_learning]]"
      - "[[information_gain]]"
---

# Epistemic Uncertainty

## Overview
Epistemic uncertainty is uncertainty due to lack of knowledge about the true model, its parameters or the state of the world. It differs from aleatoric uncertainty, which is variability inherent in the process, such as the outcome of a fair coin toss, and cannot be removed by collecting more data. Epistemic uncertainty can be reduced by gathering evidence, which is why it is also called reducible uncertainty. The distinction is a modelling convenience: what counts as irreducible noise depends on the model class and the available measurements, as discussed by Der Kiureghian and Ditlevsen (2009) and Hüllermeier and Waegeman (2021).

## Quantification
In a Bayesian treatment with parameters $\theta$, the predictive entropy splits into two parts:
$$
\underbrace{H\big[\,\mathbb{E}_{p(\theta\mid D)}\,p(y\mid x,\theta)\big]}_{\text{total}}=\underbrace{\mathbb{E}_{p(\theta\mid D)}\,H\big[p(y\mid x,\theta)\big]}_{\text{aleatoric}}+\underbrace{I(y;\theta\mid x,D)}_{\text{epistemic}}.
$$
The epistemic part is the mutual information between the prediction and the parameters, which is zero when the posterior is a point mass. In practice it is estimated from disagreement among posterior samples, ensemble members or dropout passes. Epistemic uncertainty tends to be large away from the training data and shrinks as data accumulate, whereas aleatoric uncertainty stays approximately constant. Both are useful: epistemic uncertainty flags inputs outside the training distribution and guides active learning, while aleatoric uncertainty indicates noise that no additional data will remove.

## Relation to Active Inference
In Active Inference the link is direct. Expected free energy contains an epistemic term, the expected information gain, which rewards policies expected to resolve uncertainty about hidden states and, when parameters are learned, about model parameters. This is the sense in which an agent is driven to explore: it prefers actions whose outcomes are informative, and the drive disappears once the relevant uncertainty is resolved. The aleatoric counterpart appears as the ambiguity of the likelihood mapping, which no policy can reduce. A distinction between uncertainty about states and uncertainty about parameters is often drawn within the framework, since they enter different terms of the objective.

## Related concepts
- [[uncertainty]] — the general notion of which this is one kind
- [[uncertainty_quantification]] — methods for estimating and reporting it
- [[epistemic_value]] — the reward for reducing it in Active Inference
- [[bayesian_deep_learning]] — a main source of epistemic estimates in neural networks
- [[information_gain]] — the quantity that measures its reduction
- [[belief_initialization]] — initial beliefs determine the starting epistemic uncertainty

## References
- Der Kiureghian, A., & Ditlevsen, O. (2009). Aleatory or epistemic? Does it matter? Structural Safety, 31, 105-112.
- Hüllermeier, E., & Waegeman, W. (2021). Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. Machine Learning, 110, 457-506.
- Friston, K., Rigoli, F., Ognibene, D., Mathys, C., Fitzgerald, T., & Pezzulo, G. (2015). Active inference and epistemic value. Cognitive Neuroscience, 6, 187-214.
