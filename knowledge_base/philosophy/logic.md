---
title: Logic
description: "Logic is the study of valid inference, covering deductive, inductive and abductive reasoning, formal systems, and how classical logic relates to probabilistic reasoning in Active Inference."
type: philosophy
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - logic
  - reasoning
  - epistemology
semantic_relations:
  - type: relates
    links:
      - "[[proof_theory]]"
      - "[[peircean_semiotics]]"
      - "[[bayesian_inference]]"
      - "[[active_inference]]"
aliases:
  - formal_logic
wikidata: Q8078
---

# Logic

## Overview

Logic is the systematic study of the forms of correct inference: which conclusions follow from which premises, independent of subject matter. This page uses the term in its general philosophical sense, covering formal logic as practised in mathematics and the broader theory of reasoning, and singles out the Peircean division into deduction, induction and abduction because the repo's semiotics and pragmatism pages rely on it.

## Core distinctions

- **Validity and soundness**: an argument is valid when the truth of the premises guarantees the truth of the conclusion; it is sound when it is valid and the premises are in fact true. Validity concerns form only.
- **Syntax and semantics**: a formal system specifies well-formed formulas and inference rules (syntax, studied by [[proof_theory]]), and separately assigns truth conditions via models (semantics). Soundness and completeness theorems connect the two for classical first-order logic.
- **Propositional and predicate logic**: propositional logic treats whole statements and connectives such as negation, conjunction and implication; first-order logic adds quantifiers over objects.
- **Non-classical logics**: intuitionistic logic rejects the law of excluded middle as a general principle; modal logics add necessity and possibility; paraconsistent logics tolerate contradictions without trivialising. These are choices about which inferences to license, not mere variants of notation.
- **Three modes of inference** in Peirce's scheme: deduction derives necessary consequences; induction generalises from samples; abduction proposes an explanatory hypothesis. Only deduction is truth-preserving.

## Logic and probability

Classical logic assigns propositions one of two truth values. Probabilistic reasoning generalises this: a probability of 1 or 0 recovers the logical cases, and Bayesian updating extends deduction to uncertain premises. Cox-style derivations show that, under certain consistency requirements, degrees of plausibility must obey probability rules, which is the usual argument for treating Bayesian inference as logic under uncertainty. This is an argument with assumptions, not a settled equivalence.

## Role in Active Inference

Active Inference is formulated in probabilistic rather than classical-logical terms: beliefs are distributions, inference is minimisation of free energy, and action selection follows expected free energy. The logical vocabulary still matters in three places. Perception can be read as abduction, inferring the hidden cause that best explains the observations. Generative models encode structural constraints that resemble logical rules when probabilities are near 0 or 1. Category-theoretic and type-theoretic treatments of compositional models, which are an active research direction, connect inference to formal logic more directly, but that programme is not part of the standard formulation.

## Related concepts

- [[proof_theory]] — formal derivations and their structure
- [[category_theory]] — categorical semantics relating logic to compositional models
- [[peircean_semiotics]] — Peirce's logic of signs and abduction
- [[logical_positivism]] — attempted to ground meaning in logical form and verification
- [[bayesian_inference]] — probabilistic generalisation of reasoning under uncertainty
- [[fallibilism]] — why inductive and abductive conclusions remain revisable
