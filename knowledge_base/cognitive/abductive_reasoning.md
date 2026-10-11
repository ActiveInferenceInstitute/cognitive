---
title: Abductive Reasoning
description: Inference from observations to the hypothesis that best explains them, covering Peirce's formulation, inference to the best explanation, and its relation to Bayesian model comparison in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - reasoning
  - inference
  - explanation
semantic_relations:
  - type: relates
    links:
      - "[[deductive_reasoning]]"
      - "[[inductive_reasoning]]"
      - "[[hypothesis_testing]]"
      - "[[free_energy_principle]]"
      - "[[pragmatism]]"
---

# Abductive Reasoning

## Overview
Abductive reasoning, or abduction, is inference that starts from an observation and proposes the hypothesis that would explain it. Charles Sanders Peirce introduced the term to name the process by which explanatory hypotheses are generated. A physician who sees a set of symptoms and concludes that a particular disease is present, or a detective who infers a course of events from traces at a scene, reasons abductively. The conclusion is plausible rather than certain.

## Logical form and relation to other inference
Peirce's schema runs: a surprising fact $C$ is observed; if $H$ were true, $C$ would be a matter of course; hence there is reason to suspect $H$. Compared with the other classical modes (see [[deductive_reasoning]], [[inductive_reasoning]]):
- **Deduction** derives conclusions that must hold given the premises.
- **Induction** generalises from observed cases to a rule or a future case.
- **Abduction** selects or constructs an explanation for the case at hand.

Peirce regarded the three as stages of enquiry: abduction proposes a hypothesis, deduction derives its testable consequences, and induction evaluates them against evidence (see [[scientific_method]], [[pragmatism]]). Later philosophers of science reframed the idea as inference to the best explanation (Harman, 1965; Lipton, 2004), in which candidate explanations are compared on virtues such as scope, simplicity and coherence with background knowledge. Because many hypotheses can fit the same data, abduction is non-monotonic, as new information can overturn the preferred explanation. It also raises two problems: how candidate hypotheses are generated, and how explanatory virtues should be weighed.

## Relation to Active Inference
Perception in the free energy framework is a form of inference to the best explanation of sensory data, with the generative model supplying candidate causes and the posterior expressing their relative support (see [[free_energy_principle]]). Bayesian model comparison formalises the weighing of explanations: marginal likelihood balances fit against complexity, a quantitative counterpart of preferring simple explanations that account for the data. Variational free energy approximates the negative log of this evidence. The link is therefore strong for the evaluation of explanations, and weaker for hypothesis generation, which Bayesian theory does not itself specify. Structure learning, in which models are proposed and compared, is the closest formal analogue of generation.

## Related concepts
- [[deductive_reasoning]] — derivation of consequences from a hypothesis
- [[inductive_reasoning]] — generalisation from observed cases
- [[hypothesis_testing]] — evaluation of candidate explanations
- [[pragmatism]] — the philosophical tradition of Peirce
- [[bayesian_inference]] — the probabilistic treatment of competing explanations

## References
- Harman, G. H. (1965). The inference to the best explanation. Philosophical Review, 74, 88-95.
- Lipton, P. (2004). Inference to the best explanation (2nd ed.). Routledge.
- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: A process theory. Neural Computation, 29, 1-49.
