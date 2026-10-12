---
title: Prior Distribution
description: The probability distribution expressing beliefs about unknown quantities before data are observed, its types, role in Bayes theorem, and its central place in Active Inference as priors over states, outcomes and policies.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bayesian_inference
  - probability
  - priors
semantic_relations:
  - type: relates
    links:
      - "[[likelihood_function]]"
      - "[[bayes_theorem]]"
      - "[[conjugate_priors]]"
      - "[[hyperprior]]"
      - "[[empirical_prior]]"
      - "[[prior_beliefs]]"
aliases:
  - prior_distributions
---

# Prior Distribution

## Overview
A prior distribution $p(\theta)$ encodes what is believed about an unknown quantity $\theta$ before the current data are seen. In Bayes theorem it is combined with the likelihood to give the posterior,
$$p(\theta\mid y)=\frac{p(y\mid\theta)\,p(\theta)}{p(y)}\;\propto\;p(y\mid\theta)\,p(\theta).$$
The prior can reflect earlier data, theoretical constraints or subjective judgement. Its influence shrinks as data accumulate when the likelihood is informative, but it can dominate when data are scarce.

## Kinds of prior
- **Informative priors** carry substantive knowledge, for instance a plausible range for a physical constant.
- **Weakly informative priors** rule out implausible values while remaining broad.
- **Non-informative or reference priors** aim for minimal influence. Jeffreys' prior, $p(\theta)\propto\sqrt{\det I(\theta)}$ with $I$ the Fisher information, is invariant under reparameterisation. Flat priors are not, and improper priors may yield improper posteriors.
- **Conjugate priors** belong to a family closed under updating with a given likelihood, such as the Beta prior for a Bernoulli probability, which gives closed-form posteriors.
- **Hierarchical and empirical priors** place distributions on the prior's own parameters (hyperpriors) or estimate them from data across related units.

## Interpretation and sensitivity
Interpretations differ. Subjective Bayesians treat the prior as a coherent statement of belief, while objective Bayesians seek rules that fix it from the problem's structure. Either way, good practice includes a prior predictive check, which simulates data from the prior to confirm that it implies plausible observations, and a sensitivity analysis over reasonable alternatives.

## Relation to Active Inference
Priors are central to the formalism, and they appear at several levels. A prior over initial hidden states fixes where inference begins, and prior beliefs about transitions encode the agent's model of dynamics. Preferences are expressed as a prior over outcomes, so that the agent expects and acts to realise preferred observations. A prior over policies, which may include habits learned from past choices, is combined with expected free energy in policy selection. Parameters of the likelihood and transition mappings have Dirichlet priors that are updated during learning. The framework thus treats goals, habits and learned structure as priors, a modelling decision whose empirical adequacy is assessed case by case.

## Related concepts
- [[likelihood_function]] — the data term combined with the prior
- [[bayes_theorem]] — the rule that produces the posterior
- [[conjugate_priors]] — priors that give closed-form updates
- [[hyperprior]] — priors on prior parameters
- [[empirical_prior]] — priors set from data
- [[prior_beliefs]] — the cognitive-science treatment of expectations

## References
- Jeffreys, H. (1946). An invariant form for the prior probability in estimation problems. Proceedings of the Royal Society of London A, 186, 453-461.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). Chapman & Hall/CRC.
- Bernardo, J. M., & Smith, A. F. M. (1994). Bayesian Theory. Wiley.
