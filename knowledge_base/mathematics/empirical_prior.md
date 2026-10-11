---
title: Empirical Prior
description: A prior distribution whose form or parameters are estimated from data or from higher levels of a hierarchical model, covering empirical Bayes and the empirical priors of hierarchical Active Inference models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bayesian_statistics
  - priors
  - hierarchical_models
semantic_relations:
  - type: relates
    links:
      - "[[prior_distribution]]"
      - "[[hierarchical_bayes]]"
      - "[[hyperprior]]"
      - "[[prior_beliefs]]"
      - "[[bayesian_statistics]]"
---

# Empirical Prior

## Overview
An empirical prior is a prior distribution that is not fixed in advance by the analyst but is informed by data. The term is used in two related senses. In empirical Bayes, the parameters of the prior (hyperparameters) are estimated from the observed data, for example by maximising the marginal likelihood $p(D\mid\phi)=\int p(D\mid\theta)\,p(\theta\mid\phi)\,d\theta$ over $\phi$, and the fitted prior is then used to compute posteriors. In hierarchical models, the prior on parameters at one level is determined by the level above, so it is learned jointly with the rest of the model.

## Empirical Bayes
Robbins (1956) introduced the empirical Bayes approach for problems in which many similar units are observed, so that the shared prior can be estimated from the ensemble. A standard outcome is shrinkage: each unit's estimate is pulled toward the group mean in proportion to how noisy its own data are, as in the James-Stein estimator. Efron and Hastie (2016) give a modern treatment. The method is a pragmatic approximation to a fully Bayesian hierarchy, which would place a hyperprior on $\phi$ and integrate it out. Because the data are used both to set the prior and to update it, uncertainty in $\phi$ is ignored and posterior intervals tend to be too narrow, which becomes important with few units.

## Hierarchical models
In a hierarchy, $p(\theta_1\mid\theta_2)$ plays the role of the prior for level 1 while $\theta_2$ is itself inferred. The prior is "empirical" in that it is constrained by the evidence from the level above, not by an external judgment (see [[hierarchical_bayes]]).

## Relation to Active Inference
In Active Inference and predictive coding, the term is standard for priors that are conditioned on the beliefs of higher levels of a hierarchical generative model, as described by Friston (2005): top-down predictions act as empirical priors on lower-level states, and prediction errors measure departures from them. In discrete formulations, the initial-state prior $\mathbf{D}$ and the policy prior (from habit counts) can be treated as learnable and thus empirical. This is a core feature of the framework's treatment of hierarchy, though it is a definitional use of the term rather than an additional empirical claim.

## Related concepts
- [[prior_distribution]] — the general notion of a prior
- [[hierarchical_bayes]] — models in which priors are learned from higher levels
- [[hyperprior]] — the fully Bayesian alternative to estimating hyperparameters
- [[prior_beliefs]] — the cognitive counterpart of priors
- [[D_matrix]] — prior over initial states in discrete models

## References
- Robbins, H. (1956). An empirical Bayes approach to statistics. Proceedings of the Third Berkeley Symposium on Mathematical Statistics and Probability, Vol. 1, 157-163.
- Friston, K. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society B, 360, 815-836.
- Efron, B., & Hastie, T. (2016). Computer Age Statistical Inference. Cambridge University Press.
