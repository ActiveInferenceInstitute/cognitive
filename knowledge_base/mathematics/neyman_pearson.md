---
title: Neyman-Pearson
description: The Neyman-Pearson framework for hypothesis testing, which fixes the type I error rate and maximises power, with the likelihood-ratio lemma and its contrast with Fisher's and Bayesian approaches.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistics
  - hypothesis_testing
  - decision_theory
semantic_relations:
  - type: relates
    links:
      - "[[hypothesis_testing]]"
      - "[[bayes_factors]]"
      - "[[likelihood_function]]"
      - "[[signal_detection]]"
      - "[[statistical_foundations]]"
---

# Neyman-Pearson

## Overview
The Neyman-Pearson framework, introduced by Jerzy Neyman and Egon Pearson (1933), treats hypothesis testing as a decision between a null hypothesis $H_0$ and an alternative $H_1$. A test is a rule that rejects $H_0$ for certain data. Its type I error probability $\alpha$ is the chance of rejecting a true $H_0$, its type II error probability $\beta$ is the chance of failing to reject a false $H_0$, and its power is $1-\beta$. The recipe fixes $\alpha$ in advance and seeks the test with the greatest power.

## The lemma
For simple hypotheses $H_0:\theta=\theta_0$ and $H_1:\theta=\theta_1$ with likelihoods $p(x\mid\theta_0)$ and $p(x\mid\theta_1)$, the likelihood ratio is
$$\Lambda(x)=\frac{p(x\mid\theta_1)}{p(x\mid\theta_0)}.$$
The Neyman-Pearson lemma states that the test rejecting $H_0$ when $\Lambda(x)>k$, with $k$ chosen so that the size equals $\alpha$, is most powerful among all tests of size at most $\alpha$. When the data are discrete a randomised test may be needed to attain exactly size $\alpha$. For a normal mean with known variance $\sigma^2$ and $\mu_1>\mu_0$, the ratio is increasing in the sample mean, so the test rejects when $\bar x>\mu_0+z_{1-\alpha}\sigma/\sqrt n$. For composite alternatives, a test that is most powerful against every alternative in a one-sided family is called uniformly most powerful, and such tests exist for monotone likelihood ratio families (Lehmann and Romano, 2005).

## Contrast with other approaches
Fisher's approach reports a p-value as graded evidence against $H_0$ without a specified alternative, and Neyman and Pearson stressed long-run error control. Modern practice often blends the two, which has been criticised. Bayesian approaches compare hypotheses by posterior odds, the product of prior odds and a Bayes factor, a ratio of marginal likelihoods rather than of maximised likelihoods.

## Relation to Active Inference
There is no direct link to the free energy principle. The likelihood ratio is the shared element: Bayesian model comparison in Active Inference approximates log model evidence by negative free energy, and differences of such quantities are log Bayes factors. Signal detection theory uses the likelihood ratio test as the optimal observer for detection. The error-control philosophy of Neyman and Pearson, by contrast, is not part of the Bayesian machinery used in the framework.

## Related concepts
- [[hypothesis_testing]] — the general procedure of which this is one school
- [[bayes_factors]] — the Bayesian analogue of the likelihood ratio
- [[likelihood_function]] — the quantity compared in the ratio
- [[signal_detection]] — application of likelihood-ratio decisions to perception
- [[power_analysis]] — planning samples to achieve a target power

## References
- Neyman, J., & Pearson, E. S. (1933). On the problem of the most efficient tests of statistical hypotheses. Philosophical Transactions of the Royal Society of London A, 231, 289-337.
- Lehmann, E. L., & Romano, J. P. (2005). Testing Statistical Hypotheses (3rd ed.). Springer.
- Casella, G., & Berger, R. L. (2002). Statistical Inference (2nd ed.). Duxbury.
