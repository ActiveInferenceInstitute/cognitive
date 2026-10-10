---
title: Deductive Reasoning
description: Reasoning from premises to conclusions that must be true if the premises are, how people perform on syllogisms and selection tasks, competing psychological theories, and probabilistic reinterpretations.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - reasoning
  - logic
  - dual-process
semantic_relations:
  - type: relates
    links:
      - "[[reasoning_problem_solving]]"
      - "[[probabilistic_inference]]"
      - "[[bayesian_inference]]"
---

# Deductive Reasoning

## Overview
Deductive reasoning derives conclusions that follow necessarily from premises: if the premises are true and the argument is valid, the conclusion cannot be false. Standard forms include propositional inferences such as modus ponens ("if P then Q; P; therefore Q") and modus tollens, and categorical syllogisms ("all A are B; all B are C; therefore all A are C"). It contrasts with inductive reasoning, where conclusions are only probable given the evidence.

## Findings
- **Belief bias**: people accept believable conclusions more often than unbelievable ones regardless of logical validity (Evans, Barston and Pollard, 1983).
- **Wason selection task**: asked which cards must be turned to test "if P then Q", most participants choose P and Q rather than P and not-Q (Wason, 1968). Performance improves when the rule has a familiar social or deontic content.
- **Difficulty varies by form**: modus tollens is harder than modus ponens, and syllogisms with more mental models are harder.

## Theories
- **Mental rules**: reasoning applies inference rules akin to a natural-deduction system.
- **Mental models**: reasoners construct models of possibilities in which the premises are true and test whether the conclusion holds in all of them; more models mean more errors (Johnson-Laird and Byrne, 1991).
- **Dual-process**: fast, heuristic processes often produce belief-based answers, and slower analytic processes can check validity (Evans, 2003; Evans and Stanovich, 2013). The extent to which these form two distinct systems is debated.
- **Probabilistic**: Oaksford and Chater (2007) argued that everyday reasoning is better described by Bayesian probability than by classical logic, so that conditionals express high conditional probability and the selection-task pattern reflects information-seeking.

## Role in Active Inference
Active Inference does not contain a separate logic module. Deductive inference is the limiting case of probabilistic inference in which premises are held with certainty and conditional probabilities approach zero or one, so that posterior belief in the conclusion is also near certain (see [[bayesian_inference]], [[probabilistic_inference]]). This suits the probabilistic accounts above and reframes the selection task as choosing the card that maximises expected information gain (see [[epistemic_value]], [[information_gain]]). The connection is conceptual: there are few Active Inference models of explicit deduction, and whether human logic uses approximate inference over structured generative models is still an open question. Planning as inference offers related treatments of sequential, rule-governed behaviour (see [[planning_as_inference]]).

## Related concepts
- [[decision_making]] — judgement under uncertainty
- [[executive_functions]] — control processes recruited by analytic reasoning
- [[working_memory]] — capacity limits constrain the number of models held
- [[metacognition]] — monitoring the validity of one's inferences

## References
- Wason, P. C. (1968). Reasoning about a rule. Quarterly Journal of Experimental Psychology, 20, 273-281.
- Evans, J. St. B. T., Barston, J. L., & Pollard, P. (1983). On the conflict between logic and belief in syllogistic reasoning. Memory & Cognition, 11, 295-306.
- Johnson-Laird, P. N., & Byrne, R. M. J. (1991). Deduction. Lawrence Erlbaum.
- Evans, J. St. B. T. (2003). In two minds: dual-process accounts of reasoning. Trends in Cognitive Sciences, 7, 454-459.
- Oaksford, M., & Chater, N. (2007). Bayesian Rationality: The Probabilistic Approach to Human Reasoning. Oxford University Press.
- Evans, J. St. B. T., & Stanovich, K. E. (2013). Dual-process theories of higher cognition: advancing the debate. Perspectives on Psychological Science, 8, 223-241.
