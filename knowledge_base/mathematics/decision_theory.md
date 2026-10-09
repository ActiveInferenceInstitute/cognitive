---
title: Decision Theory
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - decision_theory
  - expected_utility
  - bayesian_decision
semantic_relations:
  - type: foundation
    links:
      - "[[utility_theory]]"
      - "[[probability_theory]]"
  - type: relates
    links:
      - "[[expected_free_energy]]"
      - "[[policy_selection]]"
      - "[[game_theory]]"
---

# Decision Theory

## Overview
Decision theory is the mathematical study of choosing among actions whose consequences are uncertain. The normative core is Bayesian decision theory: given beliefs over states of the world and a numerical valuation of outcomes, choose the action with the greatest expected utility (equivalently, the smallest expected loss).

## Definition / Formalism
Let $A$ be a set of actions, $S$ a set of states with belief $p(s)$, and $u(a,s)$ the utility of taking $a$ when the state is $s$. The expected-utility rule is

$$a^* = \arg\max_{a \in A} \; \mathbb{E}_{p(s)}[u(a,s)] = \arg\max_{a} \sum_s p(s)\,u(a,s).$$

In statistical decision theory the same rule is written with a loss $L = -u$, and the expected loss under the posterior is the posterior risk; an estimator minimising it is a Bayes rule. For sequential problems the rule is applied recursively (Bellman optimality), which gives Markov decision processes and, with hidden states, POMDPs. Von Neumann and Morgenstern showed that preferences over lotteries satisfying completeness, transitivity, continuity and independence are representable by expected utility; Savage derived both probability and utility from preferences over acts. Empirical violations (the Allais and Ellsberg paradoxes) motivated descriptive alternatives such as prospect theory.

## Role in Active Inference
Active inference replaces the utility function with a prior preference distribution $\tilde p(o)$ over outcomes, with $u(o) = \ln \tilde p(o)$ up to a constant. The risk term of expected free energy is

$$D_{\mathrm{KL}}\big[q(o\mid\pi)\,\|\,\tilde p(o)\big] = -\mathbb{E}_{q(o\mid\pi)}[\ln\tilde p(o)] - H[q(o\mid\pi)],$$

so risk is expected utility (with a sign) minus an entropy bonus on predicted outcomes. Expected free energy is risk plus ambiguity, where ambiguity is the expected entropy of the likelihood. Hence active inference is not identical to expected-utility maximisation: the entropy and ambiguity terms add information-seeking behaviour that a plain expected-utility agent with fixed beliefs does not have. Policies are then drawn from a softmax over $-\gamma G(\pi)$ rather than by argmax. The two coincide only in special cases, for example when ambiguity is the same across policies and the entropy term is negligible or the predicted outcomes are deterministic.

## Related concepts
- [[utility_theory]] — supplies the valuation function that decision rules maximise
- [[expected_free_energy]] — the active inference objective that generalises expected utility
- [[policy_selection]] — how policies are scored and chosen in this repo
- [[game_theory]] — decision problems with several interacting decision makers
- [[reinforcement_learning]] — sequential decision making with learned values
- [[active_inference_pomdp]] — the POMDP setting for sequential decisions with hidden states

## References
- von Neumann & Morgenstern (1944), Theory of Games and Economic Behavior.
- Savage (1954), The Foundations of Statistics.
- Wald (1950), Statistical Decision Functions.
