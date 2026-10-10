---
title: Scientific Method
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - philosophy_of_science
  - methodology
  - inference
aliases:
  - philosophy_of_science
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[epistemic_foraging]]"
      - "[[experiment_design]]"
---

# Scientific Method

## Overview
The scientific method is not one procedure but a cluster of norms for generating, testing and revising claims about the world using evidence, public criticism and replication. Philosophy of science asks which of these norms actually justify the conclusions drawn.

## Core claims
**Hypothetico-deductive account.** Conjecture a hypothesis, deduce testable consequences, and compare them with observation. It leaves open where hypotheses come from and how to weigh partial support.

**Peirce's three inferences.** Abduction proposes an explanation, deduction derives its consequences, and induction evaluates them against data. Peirce treated the cycle as self-correcting rather than foundational, which links the method to [[pragmatism]].

**Falsificationism, paradigms, programmes.** Popper stressed refutation; Kuhn described normal science inside paradigms punctuated by revolutions; Lakatos described research programmes with protected cores and revisable auxiliary belts. Each captures part of actual practice and none is universally accepted.

**Bayesian reconstruction.** Updating credences by Bayes' rule formalizes how evidence shifts support among hypotheses, and Lindley's expected information gain formalizes which experiment to run next.

**Operational and social norms.** Defined measurement procedures ([[operationalism]]), reporting enough detail to repeat the work ([[reproducible_research]]), and adversarial peer scrutiny.

## Role in Active Inference
The analogy is old: Gregory described perception as hypothesis testing, and active inference makes the parallel quantitative. Perception is Bayesian updating of beliefs over hidden states given observations; action is the choice of observations that will discriminate between hypotheses. Selecting policies by expected free energy contains an epistemic term, the expected information gain, which is the same quantity as Bayesian optimal experimental design. In this repository's convention, G = risk + ambiguity, and ambiguity already equals H[q(o)] minus the mutual information between states and observations, so information seeking is built into the objective without a separate bonus (see [[knowledge_base/mathematics/expected_free_energy]]).

The analogy has limits. An agent that selects informative observations is an experimenter in structure only; it does not formulate new hypotheses (model structure is fixed unless structure learning is added) and it does not resist theory-ladenness, since its likelihood `A` is its observational theory.

## Related concepts
- [[pragmatism]] — Peirce's logic of inquiry
- [[operationalism]] — measurement procedures as method
- [[logical_positivism]] — the verificationist model of method
- [[verification_theory]] — confirmation and falsification
- [[epistemic_foraging]] — information-seeking action
- [[hypothesis_testing]] — statistical practice within the method
- [[reproducible_research]] — the replication norm

## References
- Popper, K. (1934). *Logik der Forschung*.
- Kuhn, T. (1962). *The Structure of Scientific Revolutions*.
- Lakatos, I. (1970). Falsification and the Methodology of Scientific Research Programmes.
- Lindley, D. V. (1956). On a measure of the information provided by an experiment. *Annals of Mathematical Statistics*.
- Gregory, R. L. (1980). Perceptions as hypotheses. *Philosophical Transactions of the Royal Society B*.
