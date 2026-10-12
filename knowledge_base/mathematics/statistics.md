---
title: Statistics
description: The discipline of collecting, summarising and drawing conclusions from data under uncertainty, covering estimation, hypothesis testing, the frequentist and Bayesian schools, and their role in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistics
  - inference
  - estimation
semantic_relations:
  - type: relates
    links:
      - "[[probability_theory]]"
      - "[[bayesian_statistics]]"
      - "[[maximum_likelihood]]"
      - "[[statistical_modeling]]"
      - "[[bayes_factors]]"
wikidata: Q12483
---

# Statistics

## Overview
Statistics is the body of methods for collecting data, describing it, and drawing conclusions about a larger population or an underlying process when observations are noisy or incomplete. It rests on [[probability_theory]], which supplies the models of how data arise, and adds procedures for going in the reverse direction, from observed data back to unknown quantities. Descriptive statistics summarise a sample, for example with means, variances and correlations. Inferential statistics quantify what the sample licenses one to say beyond itself.

## Core ideas
- **Estimation.** A parameter $\theta$ is estimated by a function of the data, an estimator. Properties such as bias, variance, consistency and efficiency are judged by repeated-sampling behaviour. Maximum likelihood chooses $\hat\theta=\arg\max_\theta p(x\mid\theta)$ and has well-understood large-sample properties.
- **Uncertainty and testing.** Confidence intervals and $p$-values describe how often a procedure would err over hypothetical repetitions of the experiment. The central limit theorem explains why normal approximations to sampling distributions are so widely useful.
- **Model comparison.** Competing models are compared with likelihood ratios, information criteria, cross-validation or Bayes factors, each trading fit against complexity differently.

## Two schools
The frequentist school defines probability as long-run frequency and treats parameters as fixed unknowns. The Bayesian school defines probability as a degree of belief and treats parameters as random variables with a prior, updated by $p(\theta\mid x)\propto p(x\mid\theta)\,p(\theta)$. The two often agree in large samples and differ in interpretation, in how prior information enters, and in how decisions are justified. Practice commonly borrows from both.

## Relation to Active Inference
The link is direct. Active Inference is built on Bayesian statistics: perception is posterior inference over hidden states, learning is inference over parameters, and the log model evidence is approximated by a variational bound (see [[variational_inference]] and [[model_evidence]]). The statistical tools above also appear in the methodology around the framework, for example in fitting models to behavioural and neuroimaging data and in comparing hypotheses by evidence. The framework is a particular theory of agents, whereas statistics is the general toolbox it draws on.

## Related concepts
- [[probability_theory]] — the formal foundation of statistical models
- [[bayesian_statistics]] — the school that Active Inference adopts
- [[maximum_likelihood]] — the standard point-estimation principle
- [[statistical_modeling]] — building and checking probabilistic models of data
- [[bayes_factors]] — evidence ratios for comparing models
- [[model_evidence]] — marginal likelihood approximated by free energy

## References
- Casella, G., & Berger, R. L. (2002). Statistical Inference (2nd ed.). Duxbury.
- Wasserman, L. (2004). All of Statistics: A Concise Course in Statistical Inference. Springer.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). Chapman and Hall/CRC.
