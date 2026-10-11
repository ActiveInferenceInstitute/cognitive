---
title: Utility Theory
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - utility
  - preferences
  - risk_aversion
semantic_relations:
  - type: foundation
    links:
      - "[[probability_theory]]"
  - type: relates
    links:
      - "[[decision_theory]]"
      - "[[pragmatic_value]]"
      - "[[expected_free_energy]]"
aliases:
  - utility_maximization
  - utility_functions
---

# Utility Theory

## Overview
Utility theory represents an agent's preferences by a real-valued function so that choosing the preferred option corresponds to maximising that function. Ordinal utility only encodes ranking; cardinal (von Neumann-Morgenstern) utility also encodes attitudes to risk because it is defined on lotteries.

## Definition / Formalism
A preference relation $\succeq$ over lotteries satisfying completeness, transitivity, continuity and independence is represented by a function $u$ such that

$$L_1 \succeq L_2 \iff \mathbb{E}_{L_1}[u] \ge \mathbb{E}_{L_2}[u].$$

The function $u$ is unique only up to a positive affine transformation $u \mapsto a u + b$ with $a>0$.

Risk attitude is read from curvature. A concave $u$ gives risk aversion: by Jensen's inequality the certainty equivalent $\mathrm{CE}=u^{-1}(\mathbb{E}[u])$ is below the expected value. The Arrow-Pratt measure is $A(x) = -u''(x)/u'(x)$. Worked example: with $u(x)=\ln x$ and a 50/50 lottery over 50 and 150, the expected value is 100 but $\mathrm{CE}=\sqrt{50\cdot150}\approx 86.60$. Multi-attribute utility, discounting, and utility from consumption streams extend this setting.

## Role in Active Inference
Active inference trades utility for a prior preference distribution over outcomes. The usual identification is $\tilde p(o)\propto \exp(u(o))$, so $\ln\tilde p(o)=u(o)+\text{const}$, and the pragmatic part of expected free energy is an expected log-preference. One difference from classical theory matters in practice: because $\tilde p$ must normalise, adding a constant to $u$ is absorbed, but multiplying $u$ by $a$ changes the distribution to $\propto\exp(a\,u)$. Scaling therefore alters how sharply preferences are held, so it is not a harmless re-parametrisation as it is for vNM utility. In the repo's discrete models this scale is carried by the preference vector $C$ (log-preferences) together with the policy precision parameter. Preferences here are beliefs about what the agent expects to observe, not exogenous rewards, which is the conceptual point of departure from reward-based reinforcement learning.

## Related concepts
- [[decision_theory]] — the choice rule that consumes a utility function
- [[pragmatic_value]] — the goal-seeking component expressed through log-preferences
- [[expected_free_energy]] — objective in which preferences enter via risk
- [[preference_learning]] — learning the preference structure from data
- [[reinforcement_learning]] — reward-based alternative to prior preferences
- [[precision_parameter]] — the inverse-temperature scale applied to policy scores

## References
- von Neumann & Morgenstern (1944), Theory of Games and Economic Behavior.
- Arrow (1971), Essays in the Theory of Risk-Bearing.
- Pratt (1964), Risk aversion in the small and in the large, Econometrica.
