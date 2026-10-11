---
title: Minimum Description Length
description: The principle that the best model of data is the one giving the shortest total code for model and data, its coding interpretation, relation to Bayesian evidence and BIC, and kinship with free energy.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - model_selection
  - compression
semantic_relations:
  - type: relates
    links:
      - "[[information_criteria]]"
      - "[[model_evidence]]"
      - "[[complexity_measures]]"
      - "[[algorithmic_information]]"
      - "[[information_theory]]"
      - "[[free_energy]]"
---

# Minimum Description Length

## Overview
The minimum description length (MDL) principle, due to Jorma Rissanen, treats learning as compression. A model is good to the extent that it allows the data to be described briefly, so the preferred hypothesis minimises the total length of two parts: the code for the model and the code for the data given the model. Regularities in data are what make compression possible, and a model that fits noise pays for its extra complexity in code length.

## Coding interpretation
By the Kraft inequality, a prefix code with codeword lengths $\ell(x)$ corresponds to a sub-probability distribution through $\ell(x)=-\log_2 p(x)$, so any probabilistic model defines a code and any code defines a model. In the two-part form,
$$L(D)=L(M)+L(D\mid M),$$
where $L(D\mid M)=-\log_2 p(D\mid M)$ measures goodness of fit. Modern versions avoid the arbitrariness of two-part codes by using a single universal code for a whole model class, such as the normalised maximum likelihood code, whose length is the maximised log-likelihood plus a complexity term that grows with the number of parameters and the sample size.

## Relation to other criteria
For a parametric model with $k$ free parameters and $n$ observations, the asymptotic code length is $-\ln p(D\mid\hat\theta)+\tfrac{k}{2}\ln n$ up to lower-order terms, which is half the Bayesian information criterion. Under a Bayesian mixture code, the code length of the data is the negative log marginal likelihood, so MDL and Bayesian evidence coincide for that code. MDL is a frequentist-flavoured and compression-based justification that does not require a prior to be believed.

## Relation to Active Inference
The connection is close and is usually stated through variational free energy. Free energy equals complexity minus accuracy, the same trade-off that MDL formalises, and in the bits-back coding argument of Hinton and van Camp the variational bound is the expected description length of the data and the latent code. Minimising free energy can therefore be read as seeking the shortest description of sensory input given the agent's model. This is an interpretive equivalence for a specified coding scheme, and it does not mean that agents literally compress data.

## Related concepts
- [[information_criteria]] — penalised-likelihood criteria that approximate MDL
- [[model_evidence]] — marginal likelihood, which equals a Bayesian code length
- [[complexity_measures]] — measures of model and data complexity
- [[algorithmic_information]] — the uncomputable ideal that MDL approximates
- [[information_gain]] — reduction in code length from learning

## References
- Rissanen, J. (1978). Modeling by shortest data description. Automatica, 14, 465-471.
- Grünwald, P. D. (2007). The Minimum Description Length Principle. MIT Press.
- Hinton, G. E., & van Camp, D. (1993). Keeping neural networks simple by minimizing the description length of the weights. Proceedings of the 6th Annual Conference on Computational Learning Theory, 5-13.
