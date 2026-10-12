---
title: Decision Analysis
description: The applied discipline that structures a choice under uncertainty into alternatives, probabilities and utilities, evaluates it with decision trees, and quantifies the value of information.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - decision_theory
  - uncertainty
  - utility
semantic_relations:
  - type: relates
    links:
      - "[[decision_theory]]"
      - "[[expected_utility]]"
      - "[[decision_making]]"
      - "[[sensitivity_analysis]]"
      - "[[utility_theory]]"
---

# Decision Analysis

## Overview
Decision analysis is a systematic procedure for making a choice under uncertainty, developed from statistical decision theory in the 1960s, notably in the work of Howard and Raiffa. Where decision theory states what a coherent decision maker would do, decision analysis supplies the working method: break a problem into parts, quantify each part, and recombine them to rank the alternatives.

## Procedure
1. **Structure the problem.** List the alternatives $a$, the uncertain events $\theta$ and the consequences. Influence diagrams and decision trees display this structure; decision nodes are chosen by the agent and chance nodes by nature.
2. **Assess probabilities.** Encode beliefs about the chance events as a distribution $p(\theta)$, using data, models or elicited judgement.
3. **Assess utilities.** Convert consequences to a scalar $u(a,\theta)$ that reflects the decision maker's preferences, including attitude to risk. With several objectives, multi-attribute utility theory combines them, often by an additive form when the attributes are suitably independent.
4. **Evaluate.** Compute $\bar u(a)=\mathbb{E}_{\theta}[u(a,\theta)]$ for each alternative; in a tree this is done by rolling back from the leaves, taking expectations at chance nodes and maxima at decision nodes.
5. **Test.** Vary probabilities and utilities to see whether the best alternative changes.

## Value of information
The expected value of perfect information is
$$\mathrm{EVPI}=\mathbb{E}_{\theta}\!\left[\max_a u(a,\theta)\right]-\max_a\mathbb{E}_{\theta}\!\left[u(a,\theta)\right]\ \ge 0,$$
which bounds what any experiment or measurement could be worth.

## Relation to Active Inference
Active Inference scores policies by expected free energy, which sums a preference-based term and an information-seeking term. The decision-analytic counterpart is expected utility plus the value of information, so the two frameworks are close in spirit. They are not identical: Active Inference derives both terms from a single objective over beliefs, while decision analysis keeps probabilities and utilities as separate elicited inputs.

## Related concepts
- [[expected_utility]] — the criterion used to rank alternatives
- [[decision_making]] — the cognitive process that decision analysis formalises
- [[dimension_reduction]] — reducing many attributes to a few when assessing consequences
- [[sensitivity_analysis]] — testing robustness of the preferred alternative
- [[expected_free_energy]] — the Active Inference analogue of a policy score
- [[cost_benefit_analysis]] — a closely related monetary evaluation method

## References
- Raiffa, H. (1968). Decision Analysis: Introductory Lectures on Choices under Uncertainty. Addison-Wesley.
- Keeney, R. L., & Raiffa, H. (1976). Decisions with Multiple Objectives: Preferences and Value Tradeoffs. Wiley.
