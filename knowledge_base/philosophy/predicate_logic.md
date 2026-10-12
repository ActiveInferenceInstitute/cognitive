---
title: Predicate Logic
description: First-order logic with quantifiers, predicates and relations, covering syntax, Tarskian semantics, the completeness and compactness theorems, undecidability, and its role in knowledge representation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - logic
  - formal_systems
  - quantification
semantic_relations:
  - type: relates
    links:
      - "[[propositional_logic]]"
      - "[[mathematical_logic]]"
      - "[[symbolic_logic]]"
      - "[[set_theory]]"
      - "[[knowledge_representation]]"
wikidata: Q35148
---

# Predicate Logic

## Overview
Predicate logic, in its standard first-order form, extends [[propositional_logic]] with variables, predicates, function symbols and the quantifiers $\forall$ ("for all") and $\exists$ ("there exists"). It can express statements about individuals and their relations, such as $\forall x\,(\mathrm{Human}(x)\to\mathrm{Mortal}(x))$, which propositional logic cannot analyse. Frege's Begriffsschrift (1879) introduced quantificational logic, and it became the standard framework for formalising mathematics.

## Syntax and semantics
A first-order language fixes a signature of predicate, function and constant symbols. Formulas are built from atomic formulas using connectives and quantifiers. A structure supplies a domain of objects together with an interpretation of each symbol, and Tarski's definition of satisfaction determines when a sentence is true in a structure. A sentence is valid if it is true in every structure, and a set of sentences entails a sentence if every structure satisfying the set satisfies it.

## Metatheory
- **Completeness.** Gödel (1930) proved that every valid first-order sentence has a formal proof, so deductive and semantic consequence coincide.
- **Compactness.** If every finite subset of a set of sentences has a model, the whole set has one. A consequence is the existence of non-standard models of arithmetic.
- **Löwenheim-Skolem.** A theory with an infinite model has models of every infinite cardinality at least as large as the language, which shows that first-order theories cannot always fix the size of their intended models.
- **Undecidability.** Church (1936) and Turing showed there is no algorithm that decides validity of arbitrary first-order sentences. Validity is, however, semi-decidable, since all valid sentences can be enumerated by proof search.

Higher-order and many-sorted logics extend the language, trading metatheoretic properties such as completeness for expressive power.

## Relation to Active Inference
The link is limited. Active Inference models are probabilistic and typically built on propositional or finite-state representations, such as the factorised discrete states of a partially observable Markov decision process. First-order structure is relevant to work in knowledge representation and to statistical relational methods that aim to combine quantification with probabilities, but the framework does not require it.

## Related concepts
- [[propositional_logic]] — the quantifier-free fragment
- [[mathematical_logic]] — the wider field in which it is studied
- [[symbolic_logic]] — the notational tradition
- [[set_theory]] — the standard first-order theory for mathematics
- [[knowledge_representation]] — a main application in artificial intelligence
- [[proof_theory]] — derivation systems for first-order logic

## References
- Enderton, H. B. (2001). A Mathematical Introduction to Logic (2nd ed.). Harcourt/Academic Press.
- Gödel, K. (1930). Die Vollständigkeit der Axiome des logischen Funktionenkalküls. Monatshefte für Mathematik und Physik, 37, 349-360.
- Church, A. (1936). A note on the Entscheidungsproblem. Journal of Symbolic Logic, 1, 40-41.
- Frege, G. (1879). Begriffsschrift, eine der arithmetischen nachgebildete Formelsprache des reinen Denkens. Louis Nebert.
