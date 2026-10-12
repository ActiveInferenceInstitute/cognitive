---
title: Expected Utility
description: The probability-weighted average of the utilities of possible outcomes, the von Neumann-Morgenstern representation theorem, common violations, and its relation to expected free energy in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - decision_theory
  - utility
  - rationality
semantic_relations:
  - type: relates
    links:
      - "[[utility_theory]]"
      - "[[decision_theory]]"
      - "[[decision_making]]"
      - "[[expected_free_energy]]"
      - "[[decision_analysis]]"
---

# Expected Utility

## Overview
Expected utility is the probability-weighted average of the utilities of the outcomes an action may produce. For an action $a$ with outcomes $x$ occurring with probability $p(x\mid a)$ and a utility function $u$,
$$\mathrm{EU}(a)=\sum_x p(x\mid a)\,u(x),$$
or the corresponding integral for continuous outcomes. The expected utility hypothesis says that a rational agent chooses the action with the highest value of $\mathrm{EU}(a)$.

## Representation theorem
Von Neumann and Morgenstern (1944) showed that if a preference relation over lotteries satisfies completeness, transitivity, continuity and independence, then there is a utility function $u$ such that one lottery is preferred to another exactly when its expected utility is higher. The function $u$ is unique up to a positive affine transformation $u\mapsto \alpha u+\beta$ with $\alpha>0$. Savage (1954) derived both subjective probabilities and utilities from preferences over acts, so that probabilities need not be given in advance. Concave $u$ corresponds to risk aversion: a sure amount is preferred to a gamble with the same mean.

## Empirical challenges
Behaviour often departs from the axioms. Allais (1953) described choice patterns that violate independence, and Ellsberg (1961) showed aversion to ambiguous probabilities. Kahneman and Tversky (1979) proposed prospect theory, in which outcomes are valued relative to a reference point and probabilities are transformed by a weighting function. These findings are descriptive rather than a refutation of the normative role of expected utility.

## Relation to Active Inference
Active Inference replaces the utility function by a prior preference over outcomes, written as a log probability, and scores policies by expected free energy. Expected free energy splits into a term that favours preferred outcomes, comparable to the expected log-preference, and an epistemic term that rewards reducing uncertainty. When the epistemic term vanishes, for example when outcomes are known given the policy, ranking policies by expected free energy reduces to ranking them by expected log-preference, which is an expected utility with $u=\ln$ preference. In general the two criteria differ because expected utility has no information-seeking term.

## Related concepts
- [[utility_theory]] — the axiomatic foundations and the forms of $u$
- [[decision_analysis]] — the applied procedure that uses expected utility to rank options
- [[decision_making]] — the cognitive process that the theory idealises
- [[expected_free_energy]] — the Active Inference score that generalises it
- [[pragmatic_value]] — the preference-seeking part of expected free energy
- [[reward_function]] — the reinforcement-learning counterpart of utility

## References
- von Neumann, J., & Morgenstern, O. (1944). Theory of Games and Economic Behavior. Princeton University Press.
- Savage, L. J. (1954). The Foundations of Statistics. Wiley.
- Kahneman, D., & Tversky, A. (1979). Prospect theory: an analysis of decision under risk. Econometrica, 47, 263-291.
- Allais, M. (1953). Le comportement de l'homme rationnel devant le risque: critique des postulats et axiomes de l'école américaine. Econometrica, 21, 503-546.
- Ellsberg, D. (1961). Risk, ambiguity, and the Savage axioms. Quarterly Journal of Economics, 75, 643-669.
