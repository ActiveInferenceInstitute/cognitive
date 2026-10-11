---
title: Model Evidence
description: The marginal likelihood of data under a model, its role in Bayesian model comparison, its automatic Occam penalty, and its approximation by negative variational free energy in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bayesian_inference
  - model_comparison
  - free_energy
semantic_relations:
  - type: relates
    links:
      - "[[bayes_factors]]"
      - "[[variational_free_energy]]"
      - "[[bayesian_inference]]"
      - "[[model_selection]]"
      - "[[information_criteria]]"
      - "[[bayesian_statistics]]"
---

# Model Evidence

## Overview
The model evidence, or marginal likelihood, is the probability of the observed data $y$ under a model $m$ after averaging over all parameter values weighted by the prior:
$$p(y\mid m)=\int p(y\mid\theta,m)\,p(\theta\mid m)\,d\theta.$$
It is the normalising constant in Bayes theorem for the parameters and the quantity used to compare models.

## Model comparison
The posterior probability of a model is proportional to its evidence times its prior probability. The ratio of the evidences of two models is the Bayes factor, which Kass and Raftery summarised with conventional verbal scales for the strength of support. The evidence penalises complexity automatically: a flexible model spreads its prior mass over many possible datasets, so it assigns a smaller probability to any single one, and it wins only if the improved fit outweighs this dilution. This behaviour is sometimes called the Bayesian Occam's razor and is shown by the decomposition of the log evidence in the large-sample limit into a maximised log-likelihood and a penalty that grows with the number of parameters. Evidence is sensitive to the prior, so conclusions can change when priors on parameters are altered.

## Approximation
The integral is usually intractable. Approximations include the Laplace method, the Bayesian information criterion, bridge and thermodynamic sampling estimators, and variational bounds. For any approximate posterior $q(\theta)$,
$$\ln p(y\mid m)=\underbrace{\mathbb{E}_q[\ln p(y,\theta\mid m)]-\mathbb{E}_q[\ln q(\theta)]}_{\text{evidence lower bound}}+\mathrm{KL}\big[q(\theta)\,\|\,p(\theta\mid y,m)\big],$$
so the lower bound equals the log evidence when $q$ is the exact posterior.

## Relation to Active Inference
The link is a derivation. Variational free energy is the negative of the evidence lower bound, so minimising it both approximates the posterior and maximises a bound on the log evidence of the agent's model. Free energy can be written as complexity minus accuracy, matching the evidence decomposition above. Perception, learning and, in extended treatments, model structure are then all described as raising evidence, which motivates the phrase self-evidencing for the behaviour of systems that seek observations their model predicts. Bayesian model reduction uses the same quantity to compare many nested priors efficiently.

## Related concepts
- [[bayes_factors]] — ratios of model evidences
- [[variational_free_energy]] — a bound on the negative log evidence
- [[information_criteria]] — approximations that penalise parameter count
- [[laplace_approximation]] — a Gaussian estimate of the evidence
- [[minimum_description_length]] — the code-length view of the same quantity
- [[model_selection]] — choosing among models using evidence

## References
- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. Journal of the American Statistical Association, 90, 773-795.
- MacKay, D. J. C. (2003). Information Theory, Inference, and Learning Algorithms, chapter 28. Cambridge University Press.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning, section 3.4. Springer.
