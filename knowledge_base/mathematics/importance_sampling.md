---
title: Importance Sampling
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - sampling
  - monte_carlo
  - estimation
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/research/concepts/monte_carlo_methods]]"
      - "[[knowledge_base/research/concepts/sampling_strategies]]"
      - "[[knowledge_base/mathematics/variational_inference]]"
      - "[[knowledge_base/mathematics/kl_divergence]]"
---

# Importance Sampling

## Overview
Importance sampling estimates an expectation under a target distribution $p$ using samples from a different proposal distribution $q$, correcting for the mismatch with weights $p/q$. It is useful when sampling from $p$ is hard but evaluating it (possibly up to a constant) is easy, and it can also estimate normalising constants such as model evidence.

## Definition / Formalism
For $q(x)>0$ wherever $p(x)f(x)\ne0$,

$$\mathbb{E}_p[f]=\mathbb{E}_q\!\left[f(x)\,w(x)\right],\quad w(x)=\frac{p(x)}{q(x)},\qquad \hat\mu=\frac1N\sum_{n=1}^N f(x_n)w(x_n),\; x_n\sim q.$$

If $p$ is known only up to a constant, $\tilde p=Zp$, the self-normalised estimator $\sum_n w_nf(x_n)/\sum_n w_n$ is consistent (slightly biased at finite $N$), and $\frac1N\sum_n\tilde w_n$ estimates $Z$. The variance is minimised by $q^*\propto|f|\,p$; for positive $f$ every term $f\,p/q^*$ is then constant (verified numerically) and the variance is zero, though $q^*$ requires knowing the answer. The effective sample size

$$N_{\mathrm{eff}}=\frac{(\sum_n w_n)^2}{\sum_n w_n^2}$$

diagnoses weight degeneracy.

Numerical check: estimating $\mathbb{E}[x^2]=1$ under $\mathcal N(0,1)$ with proposal $\mathcal N(0,2^2)$ and $2\times10^5$ draws gave 1.0015 (standard error 0.0016) with $N_{\mathrm{eff}}\approx1.3\times10^5$. A proposal with lighter tails than the target, $\mathcal N(0,0.5^2)$, dropped $N_{\mathrm{eff}}$ to about $7.9\times10^3$ and returned a visibly biased 0.871. Proposals should have heavier tails than the target.

## Role in Active Inference
Importance sampling is not used in this repo: discrete expected free energy, belief updating and policy posteriors are computed exactly from the $A,B,C,D,E$ matrices. It matters when the generative model is continuous or high-dimensional, where model evidence and expected free energy have no closed form and are approximated by sampling. The ratio $p/q$ also underlies the KL divergence ($D_{\mathrm{KL}}(q\|p)=\mathbb{E}_q[\ln q/p]$), and Jarzynski-type estimators are importance-sampling identities. The approach is therefore an alternative to variational approximation, not part of it.

## Related concepts
- [[monte_carlo_methods]] — importance sampling as one estimator family
- [[sampling_strategies]] — choosing designs and proposals
- [[variational_inference]] — the optimisation-based alternative
- [[kl_divergence]] — governs the efficiency of the best proposal
- [[bayesian_inference]] — evidence and posterior expectations as targets
- [[fluctuation_theorems]] — Jarzynski's equality as a weighted average

## References
- Owen, A. B. (2013). *Monte Carlo theory, methods and examples* (online book).
