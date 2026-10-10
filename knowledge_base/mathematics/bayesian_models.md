---
title: Bayesian Models
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - bayesian
  - cognitive_modeling
  - model_evidence
aliases:
  - bayesian_modeling
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/bayes_theorem]]"
      - "[[knowledge_base/mathematics/bayesian_generative_models]]"
      - "[[knowledge_base/cognitive/bayesian_inference]]"
      - "[[knowledge_base/cognitive/bayesian_brain]]"
---

# Bayesian Models

## Overview
A Bayesian model specifies a prior over unknown quantities and a likelihood linking them to data, and answers every question by conditioning. In cognitive science the term usually means a model in which perception, learning or decision-making is described as inference over hidden causes. This page covers the general recipe and model comparison; [[bayesian_generative_models]] covers the generative-structure side in more depth.

## Definition / Formalism
Given parameters or latent variables $\theta$, a prior $p(\theta)$ and likelihood $p(y\mid\theta)$:

$$p(\theta\mid y)=\frac{p(y\mid\theta)\,p(\theta)}{p(y)},\qquad p(y)=\int p(y\mid\theta)p(\theta)\,d\theta .$$

Prediction averages over the posterior: $p(y^*\mid y)=\int p(y^*\mid\theta)p(\theta\mid y)\,d\theta$. Models $M_1,M_2$ are compared by their evidence $p(y\mid M)$ through the Bayes factor $\mathrm{BF}_{12}=p(y\mid M_1)/p(y\mid M_2)$; the integral over $\theta$ penalises complexity automatically.

Gaussian cue combination is the canonical cognitive example. With prior $\mathcal N(\mu_0,1/\pi_0)$ and a single measurement $y$ of precision $\pi_y$, the posterior is Gaussian with

$$\pi_{\text{post}}=\pi_0+\pi_y,\qquad \mu_{\text{post}}=\frac{\pi_0\mu_0+\pi_y y}{\pi_0+\pi_y}.$$

For prior $\mathcal N(0,2^2)$ ($\pi_0=0.25$) and $y=1.5$ with unit variance, the posterior mean is 1.2 and variance 0.8, matching a grid-based numerical posterior. Less reliable cues are weighted less, which is the quantitative content behind many "Bayesian brain" claims.

Evidence example: 37 successes in 100 Bernoulli trials, comparing a fair-coin model against one with a uniform prior on the success probability, gives $\mathrm{BF}_{10}=3.67$ in favour of the flexible model, computed from $B(38,64)/0.5^{100}$.

Most realistic models lack closed-form posteriors and use approximate inference: variational methods, sampling, or Laplace approximation.

## Role in Active Inference
Active inference is a Bayesian model of an agent in which the posterior over hidden states is computed approximately by minimising variational free energy, and in which the model also carries priors over outcomes (preferences) and policies. In this repo's discrete case the posterior is exact for a single observation: `DiscreteGenerativeModel.posterior` multiplies the prior $D$ by the likelihood row $A[o,:]$ and normalises. With $A=\begin{pmatrix}0.9&0.1\\0.1&0.9\end{pmatrix}$, flat prior and $o=0$ the posterior is $(0.9,0.1)$ and the evidence is 0.5. Whether the brain actually implements Bayesian inference is an empirical claim separate from the fact that models of behaviour can be written this way.

## Related concepts
- [[bayes_theorem]] — the update rule underlying everything here
- [[bayesian_generative_models]] — model structure and latent variables
- [[bayesian_inference]] — the inference procedure
- [[bayesian_brain]] — the cognitive hypothesis
- [[information_criteria]] — BIC as an approximation to log evidence
- [[model_comparison]] — evidence-based selection in practice
