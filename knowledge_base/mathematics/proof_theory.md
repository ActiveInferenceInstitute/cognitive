---
title: Proof Theory
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - proof_theory
  - logic
  - formal_systems
semantic_relations:
  - type: relates
    links:
      - "[[type_theory]]"
      - "[[category_theory]]"
      - "[[computational_complexity]]"
---

# Proof Theory

## Overview
Proof theory treats formal proofs as mathematical objects and studies their structure: which statements are derivable in a given system, how proofs can be transformed, and what that implies about the consistency and strength of the system.

## Definition / Formalism
A formal system fixes a language, axioms, and inference rules. Common presentations are Hilbert-style systems (many axioms, modus ponens as the main rule), natural deduction (introduction and elimination rules per connective), and Gentzen's sequent calculus, where a sequent $\Gamma\vdash\Delta$ asserts that the conjunction of $\Gamma$ implies the disjunction of $\Delta$. The cut rule is

$$\frac{\Gamma\vdash A,\Delta\qquad \Gamma',A\vdash\Delta'}{\Gamma,\Gamma'\vdash\Delta,\Delta'}.$$

Gentzen's Hauptsatz states that cut is eliminable for classical and intuitionistic first-order logic: every derivable sequent has a cut-free proof. A consequence is the subformula property, which makes proof search and consistency arguments tractable. Gentzen also proved the consistency of Peano arithmetic by transfinite induction up to the ordinal $\varepsilon_0$, which is consistent with Godel's second incompleteness theorem because the argument uses methods not formalisable inside Peano arithmetic itself. Ordinal analysis assigns such ordinals to theories as a measure of strength. Under the Curry-Howard correspondence, normalising a natural-deduction proof corresponds to evaluating a typed program.

## Role in Active Inference
There is no direct role in the free energy principle. The relevance is methodological. Active inference treats reasoning as inference under a generative model, whereas proof theory characterises deduction without probabilities; relating them is the subject of probabilistic logics and of Bayesian accounts of mathematical reasoning, which are research topics rather than established results here. For this repository proof theory matters where results are formalised in a proof assistant, where claims about decidability and complexity of proof search bound what any automated reasoner, active-inference-based or not, can achieve, and where the vault discusses machine reasoning about mathematics.

## Related concepts
- [[type_theory]] — typed calculi whose terms are proofs
- [[category_theory]] — categorical semantics of proofs and logic
- [[computational_complexity]] — cost of proof search and verification
- [[ai_for_mathematics_cognitive_perspective]] — cognitive and AI views of mathematical reasoning
- [[reasoning_problem_solving]] — human reasoning contrasted with formal deduction
- [[bayes_theorem]] — the probabilistic counterpart for reasoning under uncertainty

## References
- Gentzen (1935), Untersuchungen uber das logische Schliessen.
- Godel (1931), Uber formal unentscheidbare Satze der Principia Mathematica und verwandter Systeme I.
