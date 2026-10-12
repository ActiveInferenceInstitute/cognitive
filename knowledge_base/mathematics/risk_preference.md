---
title: Risk Preference
description: An agent's attitude toward uncertain outcomes, formalised through the curvature of the utility function, risk premium and Arrow-Pratt measures, and contrasted with prospect theory and with risk in expected free energy.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - decision_theory
  - utility
  - uncertainty
semantic_relations:
  - type: relates
    links:
      - "[[expected_utility]]"
      - "[[utility_theory]]"
      - "[[decision_theory]]"
      - "[[jensen_inequality]]"
      - "[[expected_free_energy]]"
---

# Risk Preference

## Overview
Risk preference describes whether an agent prefers a gamble to a sure outcome with the same expected value. An agent that prefers the sure amount is risk averse, one indifferent between them is risk neutral, and one that prefers the gamble is risk seeking. In expected utility theory, introduced by von Neumann and Morgenstern, these attitudes are properties of the utility function $u$ over outcomes and not separate parameters.

## Formal treatment
For a random outcome $X$ the agent values the lottery at $\mathbb{E}[u(X)]$. By Jensen's inequality, if $u$ is concave then $\mathbb{E}[u(X)]\le u(\mathbb{E}[X])$, so the agent is risk averse; convex $u$ gives risk seeking; linear $u$ gives risk neutrality. The certainty equivalent $c$ solves $u(c)=\mathbb{E}[u(X)]$, and the risk premium is $\mathbb{E}[X]-c$, which is positive for a risk-averse agent. For small risks, Pratt (1964) showed that the premium is approximately
$$\tfrac12\,\mathrm{Var}(X)\,A(x),\qquad A(x)=-\frac{u''(x)}{u'(x)},$$
where $A$ is the Arrow-Pratt coefficient of absolute risk aversion. Constant $A$ corresponds to exponential utility, and a decreasing $A$ means that wealthier agents accept more risk.

Experiments show systematic departures from expected utility. In prospect theory, Kahneman and Tversky (1979) proposed that people evaluate gains and losses relative to a reference point, with a value function that is concave for gains and convex for losses, steeper for losses than for gains, and with distorted probability weights. This predicts risk aversion for moderate-probability gains and risk seeking for losses, a pattern observed in many studies, although its generality and the size of the effects depend on the task.

## Relation to Active Inference
In expected free energy, "risk" has a different technical meaning. It denotes the divergence between predicted outcomes under a policy and preferred outcomes, $\mathrm{KL}[q(o\mid\pi)\,\|\,p(o)]$, and is paired with ambiguity, the expected uncertainty of the observation mapping. Minimising this term yields pragmatic behaviour that moves predicted outcomes toward preferred ones. It is not the economic notion of aversion to outcome variance, although a precise prior preference can make an agent avoid outcome variability. Whether human risk preferences are best modelled by prior preferences in this sense is an empirical question under study.

## Related concepts
- [[expected_utility]] — the decision rule in which risk attitude is encoded
- [[utility_theory]] — foundations of preference representation
- [[decision_theory]] — the wider field of choice under uncertainty
- [[jensen_inequality]] — links concavity of utility to risk aversion
- [[risk_sensitive_control]] — control criteria that penalise outcome variance
- [[decision_making]] — behavioural and cognitive studies of choice

## References
- Pratt, J. W. (1964). Risk aversion in the small and in the large. Econometrica, 32, 122-136.
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. Econometrica, 47, 263-291.
- Mas-Colell, A., Whinston, M. D., & Green, J. R. (1995). Microeconomic Theory. Oxford University Press.
