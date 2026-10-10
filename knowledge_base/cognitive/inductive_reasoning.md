---
title: Inductive Reasoning
description: Inductive reasoning generalizes from observed instances to conclusions that go beyond them, and this page covers its psychology, Bayesian formalization, and relation to Active Inference.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - reasoning
  - generalization
  - bayesian
semantic_relations:
  - type: relates
    links:
      - "[[reasoning_problem_solving]]"
      - "[[bayesian_inference]]"
      - "[[structure_learning]]"
---

# Inductive Reasoning

## Overview
Inductive reasoning draws conclusions that extend beyond the evidence: from some observed cases to unobserved ones, or from past regularities to future ones. Unlike deduction, a correct inductive inference can still have a false conclusion, so its strength is graded rather than guaranteed. In cognitive science the term covers category-based generalization ("robins have property P, so sparrows probably do"), causal and rule induction, and analogical projection.

## Definition / Formalism
Philosophically, the problem of induction (Hume) is that no deductive argument justifies expecting the future to resemble the past. Psychology sidesteps justification and asks which generalizations people make and why. Consistent empirical findings include similarity effects (the more similar the premise and conclusion categories, the stronger the inference), diversity effects (varied premises support a broader conclusion than redundant ones), and sensitivity to sample size and how examples were selected.

Bayesian models treat induction as inference over hypotheses. With a hypothesis space $\mathcal{H}$, prior $P(h)$, and observed examples $X$, the posterior is $P(h \mid X) \propto P(X \mid h)\,P(h)$, and a new case is accepted in proportion to the posterior mass of hypotheses that include it. If examples are sampled from the true concept (strong sampling), the likelihood of $n$ examples under a hypothesis covering $|h|$ items is $|h|^{-n}$. This "size principle" favors the smallest consistent hypothesis and reproduces the sharp generalization seen after few examples. Which hypothesis space and prior people use is the main modelling choice.

## Role in Active Inference
Induction maps onto learning a generative model. Updating beliefs about parameters from data is Bayesian inference over hypotheses, and in discrete-state Active Inference this appears as Dirichlet count updating. Structure learning, in which an agent compares or grows model structures through Bayesian model comparison and reduction, is the closest analogue of choosing among competing hypothesis spaces. The expected free energy's epistemic term also gives a principled reason to seek the observations that most discriminate between competing generalizations. The correspondence with the human-subjects literature is a modelling alignment, not a demonstrated mechanism: Active Inference supplies the inference engine, while the hypothesis spaces and priors that explain specific inductive judgments must still be specified for each task.

## Related concepts
- [[reasoning_problem_solving]] — the broader family of reasoning types, including deduction and abduction
- [[bayesian_inference]] — the normative framework used to model induction
- [[structure_learning]] — learning which model structure explains the data
- [[statistical_learning]] — implicit induction of regularities from exposure
- [[hypothesis_testing]] — formal procedures for evaluating generalizations
- [[epistemic_value]] — the information-seeking term that favors discriminating observations

## References
- Tenenbaum, J. B., Kemp, C., Griffiths, T. L., & Goodman, N. D. (2011). How to grow a mind: Statistics, structure, and abstraction. *Science*, 331, 1279-1285.
