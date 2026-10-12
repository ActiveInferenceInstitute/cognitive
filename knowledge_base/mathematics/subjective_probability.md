---
title: Subjective Probability
description: The interpretation of probability as a rational agent's degree of belief, with its coherence arguments, representation theorems, criticisms, and its role in Bayesian decision theory and Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - probability_interpretation
  - bayesian
  - decision_theory
semantic_relations:
  - type: relates
    links:
      - "[[probability_theory]]"
      - "[[bayesian_statistics]]"
      - "[[decision_theory]]"
      - "[[expected_utility]]"
      - "[[belief_state]]"
---

# Subjective Probability

## Overview
In the subjective interpretation, a probability is a degree of belief held by an agent, not a long-run frequency or a physical propensity. Different agents with different information may assign different probabilities to the same event without either being mistaken, and probability statements are constrained by coherence rather than by matching a unique external quantity. Ramsey (1926, published 1931) and de Finetti (1937) developed the view independently, and Savage (1954) combined it with utility theory.

## Justifications
- **Coherence and Dutch books.** Degrees of belief can be elicited through the betting rates an agent considers fair. If those rates violate the axioms of probability, for example by assigning to mutually exclusive events probabilities that do not add correctly, a bookmaker can arrange bets that guarantee the agent a loss. The converse theorem states that coherent rates satisfy the probability axioms, with finite additivity in de Finetti's treatment.
- **Representation theorems.** Savage showed that if an agent's preferences among acts satisfy a set of axioms, there is a probability measure on states and a utility function, unique up to a positive affine transformation, such that preferences agree with expected utility. Probability is thereby derived from choice behaviour ([[expected_utility]], [[utility_theory]]).
- **Updating.** Conditioning on evidence by [[bayes_theorem]] is the rule for revising degrees of belief, and it is supported by diachronic coherence arguments.

## Debates
Critics note that experimental findings, including the Ellsberg (1961) paradox on ambiguity aversion, show systematic departures from the Savage axioms in human choice. Objective Bayesians such as Jaynes argue that principles like maximum entropy should constrain priors, so that probabilities are not wholly personal ([[maximum_entropy_principle]]). Frequentists reject the use of probability for single hypotheses. These are open disagreements about foundations. Many practising statisticians use both interpretations according to the problem.

## Relation to Active Inference
Active Inference adopts a Bayesian view in which an agent's beliefs about hidden states, parameters and policies are probability distributions, i.e. subjective probabilities encoded by the agent's internal states ([[belief_state]], [[belief_updating]]). Prior preferences over outcomes play the role of utilities in a formulation where preferences are expressed as probabilities, so that the distinction between belief and desire is formal. It is a modelling stance about how to describe agents, not a claim that organisms hold explicit numerical probabilities, and the status of Dutch-book style rationality for biological agents is debated.

## Related concepts
- [[probability_theory]] — the formal calculus beliefs must satisfy
- [[bayesian_statistics]] — statistical practice built on the interpretation
- [[decision_theory]] — probability combined with utility
- [[expected_utility]] — the criterion in Savage's theorem
- [[belief_state]] — the agent's distribution over states
- [[prior_distribution]] — the starting beliefs before evidence

## References
- Ramsey, F. P. (1931). Truth and probability. In R. B. Braithwaite (Ed.), The Foundations of Mathematics and Other Logical Essays. Kegan Paul, Trench, Trubner & Co.
- Savage, L. J. (1954). The Foundations of Statistics. Wiley.
- de Finetti, B. (1974). Theory of Probability: A Critical Introductory Treatment (Vol. 1). Wiley.
- Ellsberg, D. (1961). Risk, ambiguity, and the Savage axioms. Quarterly Journal of Economics, 75, 643-669.
- de Finetti, B. (1937). La prévision: ses lois logiques, ses sources subjectives. Annales de l'Institut Henri Poincaré, 7, 1-68.
