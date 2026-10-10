---
title: Set Theory
description: "Sets, relations, functions, cardinality and the ZFC axioms as the standard foundation of mathematics, with notes on its role in probability, category theory and philosophy of mathematics."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - foundations
  - logic
  - sets
semantic_relations:
  - type: relates
    links:
      - "[[category_theory]]"
      - "[[type_theory]]"
      - "[[philosophy_of_mathematics]]"
      - "[[measure_theory]]"
---

# Set Theory

## Overview
Set theory studies collections of objects and the operations on them. Introduced by Cantor in the 1870s, it was axiomatised in the early twentieth century to avoid paradoxes such as Russell's ($R=\{x: x\notin x\}$), and it now serves as the default foundation in which most mathematics can be encoded.

## Core content
**Basic notions.** Membership, subset, union, intersection, complement, power set $\mathcal P(X)$, Cartesian product, relations (equivalence relations correspond to partitions), and functions as sets of ordered pairs.

**Cardinality.** Two sets have equal cardinality if a bijection exists. Cantor's theorem shows $|X|<|\mathcal P(X)|$, so there are uncountably many reals and no largest infinity. Whether any cardinal lies strictly between $|\mathbb N|$ and $|\mathbb R|$ (the continuum hypothesis) is independent of ZFC (Godel 1940; Cohen 1963).

**ZF and ZFC axioms.** Extensionality, pairing, union, power set, infinity, foundation (regularity), and the separation and replacement schemas (together ZF); adding the axiom of choice gives ZFC. Choice is equivalent to the well-ordering theorem and Zorn's lemma, and is used to prove that every vector space has a basis.

**Ordinals.** Well-ordered sets extending counting into the transfinite, supporting induction and recursion beyond $\omega$.

**Alternatives.** [[category_theory]] and [[type_theory]] offer different foundations in which structure and constructions, not membership, are primitive.

## Role in Active Inference
Set theory enters as language rather than method. State spaces, observation spaces and policy sets are sets; a probability space $(\Omega,\mathcal F,P)$ is built from sets and a $\sigma$-algebra ([[measure_theory]]); Markov blanket definitions partition a set of states into internal, external, sensory and active subsets. None of the Active Inference equations depend on a choice of foundation. The set-theoretic versus categorical presentation is a matter of taste, though compositional treatments of Active Inference (see [[category_theory]]) prefer the latter. Philosophical questions, such as whether mathematical objects exist independently of minds, belong to [[philosophy_of_mathematics]].

## Related concepts
- [[category_theory]] — structure-first alternative to membership
- [[measure_theory]] — $\sigma$-algebras are families of sets
- [[proof_theory]] — the formal systems in which ZFC is stated
- [[probability_theory]] — events as subsets
