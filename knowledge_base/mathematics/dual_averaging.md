---
title: Dual Averaging
description: Dual averaging is a first-order method for convex optimisation that minimises the running sum of past gradients plus a regulariser, and is used for regularised learning and for step-size adaptation in Hamiltonian Monte Carlo.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimization
  - convex_optimization
  - sampling
semantic_relations:
  - type: relates
    links:
      - "[[gradient_descent]]"
      - "[[stochastic_optimization]]"
      - "[[hamiltonian_monte_carlo]]"
      - "[[optimization_theory]]"
      - "[[path_integral_implementation]]"
---

# Dual Averaging

## Overview
Dual averaging, introduced by Yurii Nesterov, is a subgradient method for convex optimisation. Ordinary gradient descent moves the current iterate against the latest gradient. Dual averaging instead keeps the sum of all past gradients, which lives in the dual space, and sets each new iterate by minimising a linear model built from that sum together with a strongly convex regulariser centred at a starting point. Every past gradient therefore has equal weight in the update.

## Update rule
Let $g_s$ be a subgradient of the objective at iterate $x_s$, let $h$ be a strongly convex prox function with minimiser $x_1$ over the feasible set $X$, and let $\beta_t$ be a non-decreasing sequence of positive weights. The iterates are
$$x_{t+1}=\arg\min_{x\in X}\Big\{\Big\langle \sum_{s=1}^{t} g_s,\,x\Big\rangle+\beta_t\,h(x)\Big\}.$$
For a convex problem with bounded subgradients the choice $\beta_t\propto\sqrt{t}$ gives average regret of order $1/\sqrt{T}$, and the averaged iterate converges at the same rate. This matches the optimal rate for general nonsmooth convex problems in the subgradient setting. Regularised dual averaging, due to Xiao, extends the method to objectives with a non-smooth term such as an $\ell_1$ penalty, and keeps the exact regulariser in each step instead of linearising it. This can produce sparse iterates more reliably than a plain subgradient step.

## Step-size adaptation in sampling
The same name is used in the No-U-Turn sampler, a variant of Hamiltonian Monte Carlo. There, dual averaging serves as a stochastic-approximation scheme for the step size $\epsilon$. After each trajectory the deviation of the acceptance statistic from a target value is averaged over iterations, and the average is mapped to a new $\log\epsilon$. This yields a step size near the target acceptance rate during warm-up, after which $\epsilon$ is fixed so that the chain remains a valid Markov chain.

## Relation to Active Inference
The connection is indirect. Dual averaging is general optimisation and sampling infrastructure and is not part of the Active Inference formalism. Its relevance here lies in implementations of path-integral and Hamiltonian-based inference, where gradient-based samplers need tuned step sizes, and in learning methods that minimise free energy with stochastic gradients.

## Related concepts
- [[gradient_descent]] — the primal method that dual averaging reweights
- [[stochastic_optimization]] — the noisy-gradient setting where averaging is useful
- [[hamiltonian_monte_carlo]] — the sampler whose step size is adapted
- [[optimization_theory]] — convergence and regret analysis
- [[learning_rate]] — the step-size parameter that is being tuned
- [[path_integral_implementation]] — a numerical setting in which adaptive step sizes matter

## References
- Nesterov, Y. (2009). Primal-dual subgradient methods for convex problems. Mathematical Programming, 120, 221-259.
- Xiao, L. (2010). Dual averaging methods for regularized stochastic learning and online optimization. Journal of Machine Learning Research, 11, 2543-2596.
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15, 1593-1623.
