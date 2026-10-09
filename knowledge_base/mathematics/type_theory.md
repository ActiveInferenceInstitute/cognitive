---
title: Type Theory
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - type_theory
  - lambda_calculus
  - formal_methods
semantic_relations:
  - type: relates
    links:
      - "[[proof_theory]]"
      - "[[category_theory]]"
      - "[[probabilistic_programming]]"
---

# Type Theory

## Overview
Type theory is a family of formal systems in which every term has a type and computation is by reduction of terms. It serves both as a foundation for mathematics, alternative to set theory, and as the theory behind typed programming languages and proof assistants.

## Definition / Formalism
The simply typed lambda calculus has types $A,B ::= \iota \mid A\to B$ and terms $M ::= x \mid \lambda x{:}A.\,M \mid M\,N$, with judgments $\Gamma\vdash M:A$ and the rules

$$\frac{\Gamma,x{:}A\vdash M:B}{\Gamma\vdash \lambda x{:}A.\,M : A\to B}\qquad \frac{\Gamma\vdash M:A\to B\quad \Gamma\vdash N:A}{\Gamma\vdash M\,N : B}.$$

Well-typed terms in this system normalise (terminate). The Curry-Howard correspondence identifies types with propositions and terms with proofs: $A\to B$ is implication, product types are conjunction, sum types are disjunction. Dependent types let a type depend on a term, so a type like "vectors of length $n$" can be expressed and checked. Martin-Lof type theory, the calculus of constructions, and homotopy type theory extend this with identity types, universes and, in the last case, the univalence axiom. Proof assistants such as Lean and Coq are implementations of dependent type theories.

## Role in Active Inference
The connection is indirect. Three places where it touches this repo's themes: (1) formalising mathematics, including machine-checked proofs of results about inference, relies on dependent type theory; (2) probabilistic programming languages use types to separate deterministic from random computation, which matters when a generative model is written as a program; (3) the compositional, category-theoretic treatment of generative models and Bayesian inference is naturally phrased in type-theoretic terms. As a lightweight analogy only, the shape constraints on the $A$, $B$, $C$, $D$ matrices of a discrete generative model are dimension checks of the kind dependent types make static; the repo enforces them at run time. Type theory is not part of the free energy principle itself.

## Related concepts
- [[proof_theory]] — the logical side of the Curry-Howard correspondence
- [[category_theory]] — typed lambda calculi are internal languages of categories
- [[probabilistic_programming]] — typed languages for generative models
- [[ai_for_mathematics_cognitive_perspective]] — machine reasoning about formal mathematics
- [[computational_complexity]] — what can be decided about typed programs

## References
- Church (1940), A formulation of the simple theory of types, Journal of Symbolic Logic.
- Martin-Lof (1984), Intuitionistic Type Theory.
- The Univalent Foundations Program (2013), Homotopy Type Theory: Univalent Foundations of Mathematics.
