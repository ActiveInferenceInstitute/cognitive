---
title: Heuristics
description: Simple rules of thumb that allow fast judgement and choice with limited information, covering the heuristics-and-biases and fast-and-frugal traditions and their relation to approximate inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - judgement
  - decision
  - bounded_rationality
semantic_relations:
  - type: relates
    links:
      - "[[decision_making]]"
      - "[[reasoning_problem_solving]]"
      - "[[cognitive_phenomena]]"
      - "[[bayesian_inference]]"
aliases:
  - heuristic
wikidata: Q201413
---

# Heuristics

## Overview
A heuristic is a rule of thumb that produces a judgement or decision quickly by using only part of the available information. Heuristics are contrasted with procedures that weigh all relevant evidence, such as expected-utility maximisation or full Bayesian updating. The study of heuristics rests on the observation that human cognition is limited in time, knowledge and computational capacity, so that satisfactory solutions are often preferred to optimal ones. Simon (1955) described this as bounded rationality.

## Two research traditions
- **Heuristics and biases.** Tversky and Kahneman (1974) described heuristics such as representativeness, availability and anchoring-and-adjustment. Judging probability by similarity to a prototype, by ease of recall, or by adjusting from an initial value usually works, but yields systematic errors in some circumstances, for example neglect of base rates. This programme treats heuristics mainly as sources of bias relative to normative standards.
- **Fast-and-frugal heuristics.** Gigerenzer and colleagues emphasise that simple rules can be accurate in structured environments, sometimes outperforming complex models when data are scarce or noisy. Examples include the recognition heuristic and take-the-best, which ranks options by the first cue that discriminates between them. In this view, the question is the ecological fit between a rule and the environment in which it is used (Gigerenzer & Gaissmaier, 2011).

The traditions disagree about whether heuristics are best evaluated against logical and probabilistic norms or against performance in realistic settings. Dual-process theories add that fast, automatic processes producing intuitive answers can be monitored by slower deliberation.

## Relation to Active Inference
Heuristics can be seen as approximations to exact inference. Exact Bayesian inference is intractable in general, so any biological agent must use approximate schemes, and variational inference is one principled approach that trades accuracy for computational cost (see [[bayesian_inference]]). Within Active Inference, habits and learned priors over policies play the role of cached shortcuts, and the cost of computation is part of what bounded agents must manage. Some work has argued that specific biases follow from reasonable priors or from approximate posteriors. This is an interpretive programme with some modelling support, and the mapping from particular heuristics to particular approximations is mostly not established.

## Related concepts
- [[decision_making]] — the setting in which heuristics are applied
- [[reasoning_problem_solving]] — heuristic search in problem solving
- [[hypothesis_testing]] — normative inference against which biases are judged
- [[cognitive_phenomena]] — related effects and biases
- [[policy_selection]] — how Active Inference scores alternative courses of action

## References
- Simon, H. A. (1955). A behavioral model of rational choice. Quarterly Journal of Economics, 69, 99-118.
- Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. Science, 185, 1124-1131.
- Gigerenzer, G., & Gaissmaier, W. (2011). Heuristic decision making. Annual Review of Psychology, 62, 451-482.
