---
title: Risk-Sensitive Control
description: Stochastic optimal control with an exponential-of-cost criterion that penalises variability of cost, its link to differential games and relative entropy, and its relation to KL control and Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimal_control
  - stochastic_control
  - risk
semantic_relations:
  - type: relates
    links:
      - "[[stochastic_control]]"
      - "[[optimal_control]]"
      - "[[path_integral_control]]"
      - "[[kl_divergence]]"
      - "[[risk_preference]]"
---

# Risk-Sensitive Control

## Overview
Standard stochastic optimal control minimises the expected total cost $\mathbb{E}[C]$, which ignores how much the cost varies between realisations. Risk-sensitive control instead minimises an exponential criterion,
$$J_\theta=\frac1\theta\ln\mathbb{E}\big[e^{\theta C}\big],$$
where $\theta\neq0$ is a risk parameter. Jacobson (1973) introduced this criterion for linear systems, and Whittle (1981, 1990) developed it for linear-quadratic-Gaussian problems and the general partially observed case.

## Properties
Expanding in cumulants gives
$$J_\theta=\mathbb{E}[C]+\tfrac{\theta}{2}\,\mathrm{Var}(C)+O(\theta^2),$$
so $\theta>0$ penalises variance and yields a risk-averse controller, $\theta<0$ rewards variance and yields a risk-seeking one, and the limit $\theta\to0$ recovers the risk-neutral case. The exponential form is chosen largely because it preserves dynamic programming: the value function satisfies a multiplicative Bellman recursion, and for linear dynamics with quadratic cost and Gaussian noise the solution is a Riccati equation with a modified gain. For $\theta>0$ the problem can fail to have a finite solution if $\theta$ is too large, a threshold effect not present in the risk-neutral problem.

Two interpretations connect the criterion to other fields. The Donsker-Varadhan variational formula gives
$$\ln\mathbb{E}_P\big[e^{\theta C}\big]=\sup_{Q}\big\{\theta\,\mathbb{E}_Q[C]-\mathrm{KL}(Q\,\|\,P)\big\},$$
so for $\theta>0$ and a cost to be minimised, risk-sensitive control is equivalent to a game against an adversary who may alter the noise distribution at a relative-entropy price. Jacobson related the criterion to deterministic differential games, a connection later linked to $H_\infty$ control. The same identity underlies KL control and linearly solvable formulations (Todorov, 2009), in which the control cost is a KL divergence from passive dynamics and optimal control reduces to a path integral (see [[path_integral_control]]).

## Relation to Active Inference
Risk-sensitive and KL-control formulations are close in form to Active Inference, since both combine an expected cost with a divergence between controlled and reference path distributions, and the "risk" term in expected free energy is a KL divergence of this kind. Derivations that connect them show that, under restricted assumptions, an Active Inference agent's behaviour can be mapped to a KL-control or risk-sensitive problem. The mapping is not general, because expected free energy includes epistemic terms that these criteria lack, so the relationship is a partial correspondence and an active topic of study.

## Related concepts
- [[stochastic_control]] — the problem class being refined
- [[optimal_control]] — the risk-neutral baseline
- [[path_integral_control]] — control as inference over paths
- [[kl_divergence]] — the penalty in the variational form
- [[risk_preference]] — attitudes to variability of outcomes
- [[hamilton_jacobi_bellman]] — equation satisfied by the value function

## References
- Jacobson, D. H. (1973). Optimal stochastic linear systems with exponential performance criteria and their relation to deterministic differential games. IEEE Transactions on Automatic Control, 18, 124-131.
- Whittle, P. (1981). Risk-sensitive linear/quadratic/Gaussian control. Advances in Applied Probability, 13, 764-777.
- Whittle, P. (1990). Risk-Sensitive Optimal Control. Wiley.
- Todorov, E. (2009). Efficient computation of optimal actions. Proceedings of the National Academy of Sciences, 106, 11478-11483.
