---
title: Acquisition Functions
description: Scoring rules used in Bayesian optimization and active learning to choose the next point to evaluate, trading off exploration and exploitation under a probabilistic surrogate model.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimization
  - bayesian_optimization
  - exploration
semantic_relations:
  - type: relates
    links:
      - "[[bayesian_optimization]]"
      - "[[gaussian_processes]]"
      - "[[exploration_exploitation]]"
      - "[[epistemic_value]]"
      - "[[expected_free_energy]]"
---

# Acquisition Functions

## Overview
An acquisition function $\alpha(x)$ assigns a score to each candidate input $x$ using the current posterior of a surrogate model, and the next point evaluated is $x_{n+1}=\arg\max_x \alpha(x)$. The objective is typically expensive, noisy or lacks gradients, so the surrogate (most often a Gaussian process) is used to decide where a costly evaluation will be most useful. Because the score depends on both the posterior mean $\mu(x)$ and the posterior standard deviation $\sigma(x)$, it encodes a trade-off between exploiting regions that look good and exploring regions that are uncertain.

## Common forms
For maximisation with current best observed value $f^{*}$ and $z=(\mu(x)-f^{*})/\sigma(x)$:
- **Probability of improvement**: $\mathrm{PI}(x)=\Phi(z)$, the posterior probability of beating $f^{*}$. It ignores the size of the improvement and can be overly greedy.
- **Expected improvement**: $\mathrm{EI}(x)=\mathbb{E}[\max(f(x)-f^{*},0)]=(\mu-f^{*})\Phi(z)+\sigma\,\phi(z)$, where $\Phi$ and $\phi$ are the standard normal distribution and density functions. It is the most widely used default.
- **Upper confidence bound**: $\mathrm{UCB}(x)=\mu(x)+\kappa\,\sigma(x)$, with $\kappa$ controlling the amount of exploration.
- **Thompson sampling**: draw one function from the posterior and maximise it, so exploration arises from posterior randomness.
- **Information-based rules**: criteria such as entropy search select points that most reduce uncertainty about the location or value of the optimum.

Each rule is cheap to evaluate compared with the objective, so the inner maximisation over $x$ is carried out with a conventional optimiser.

## Relation to Active Inference
The link is a structural analogy. In Active Inference, the expected free energy of a policy splits into an epistemic term (expected information gain) and a pragmatic term (expected preference satisfaction), and selecting policies by this sum produces behaviour that mixes exploration and exploitation without a separate exploration parameter. An acquisition function plays the same role for a Bayesian optimiser: it scores candidate actions by expected utility plus expected uncertainty reduction. The information-based acquisition rules are the closest match to epistemic value. The two are not equivalent, since the preference term in Active Inference is a prior over outcomes, whereas improvement-based acquisition functions measure gain over the best value seen so far.

## Related concepts
- [[bayesian_optimization]] — the sequential design procedure that uses acquisition functions
- [[gaussian_processes]] — the usual surrogate supplying $\mu(x)$ and $\sigma(x)$
- [[exploration_exploitation]] — the trade-off the score balances
- [[epistemic_value]] — information-seeking component of Active Inference
- [[multi_armed_bandits]] — discrete setting where UCB and Thompson sampling originated

## References
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. Journal of Global Optimization, 13, 455-492.
- Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the human out of the loop: A review of Bayesian optimization. Proceedings of the IEEE, 104, 148-175.
- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.
