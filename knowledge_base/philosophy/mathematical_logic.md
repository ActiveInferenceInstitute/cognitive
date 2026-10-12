---
title: Mathematical Logic
description: The mathematical study of formal languages, proofs and models, covering its main branches, the completeness and incompleteness theorems, and its limited but real connections to probability and computation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - logic
  - formal_systems
  - foundations
semantic_relations:
  - type: relates
    links:
      - "[[propositional_logic]]"
      - "[[predicate_logic]]"
      - "[[proof_theory]]"
      - "[[set_theory]]"
      - "[[symbolic_logic]]"
---

# Mathematical Logic

## Overview
Mathematical logic applies mathematical methods to the study of formal languages, deduction and mathematical structures. It grew out of nineteenth-century work by Boole and Frege and of early twentieth-century concern with the foundations of mathematics. The field is usually divided into four areas: model theory, proof theory, recursion (computability) theory and set theory.

## Main branches
- **Model theory** studies the relation between formal sentences and the structures that satisfy them. Its basic notion is the satisfaction of a sentence in a structure, as defined by Tarski.
- **Proof theory** treats proofs as mathematical objects and asks which statements a given system can derive, and by what normal forms (see [[proof_theory]]).
- **Computability theory** classifies problems by whether an algorithm can decide them. Church and Turing showed in 1936 that validity in first-order logic is undecidable.
- **Set theory** supplies an axiomatic framework, usually Zermelo-Fraenkel set theory with choice, within which most of ordinary mathematics can be formalised (see [[set_theory]]).

## Landmark results
Gödel's completeness theorem (1930) shows that a first-order sentence is derivable from a set of axioms exactly when it holds in every model of them. His incompleteness theorems (1931) show that any consistent, effectively axiomatised theory containing enough arithmetic contains true statements it cannot prove, and cannot prove its own consistency. Together these results fixed the limits of formal axiomatisation as a foundation, and they remain central to philosophical discussion of what mathematical knowledge is.

## Relation to Active Inference
There is no derivation of Active Inference from mathematical logic. The connection is indirect. Probability theory can be read as an extension of logic to uncertain propositions, a view developed by Jaynes, and Bayesian inference is then a generalisation of deduction. Formal logic also underlies the specification languages used to describe agents and their goals. These links concern the mathematics available to the framework, not claims about how brains reason, and logic itself plays no central role in the free energy principle.

## Related concepts
- [[propositional_logic]] — the simplest formal system studied
- [[predicate_logic]] — first-order logic, the main setting for model theory
- [[proof_theory]] — formal derivations as objects of study
- [[set_theory]] — the usual axiomatic foundation
- [[symbolic_logic]] — the symbolic notation the field uses
- [[modal_logic]] — extensions for necessity and possibility

## References
- Enderton, H. B. (2001). A Mathematical Introduction to Logic (2nd ed.). Harcourt/Academic Press.
- Gödel, K. (1931). Über formal unentscheidbare Sätze der Principia Mathematica und verwandter Systeme I. Monatshefte für Mathematik und Physik, 38, 173-198.
- Jaynes, E. T. (2003). Probability Theory: The Logic of Science. Cambridge University Press.
