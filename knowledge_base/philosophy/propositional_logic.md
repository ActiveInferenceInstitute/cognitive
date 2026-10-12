---
title: Propositional Logic
description: The logic of whole statements combined by truth-functional connectives, covering syntax, truth tables, validity, deductive systems, completeness, and the computational difficulty of satisfiability.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - logic
  - formal_systems
  - deduction
semantic_relations:
  - type: relates
    links:
      - "[[predicate_logic]]"
      - "[[mathematical_logic]]"
      - "[[symbolic_logic]]"
      - "[[deductive_reasoning]]"
      - "[[syllogistic_logic]]"
---

# Propositional Logic

## Overview
Propositional logic, also called sentential logic, studies the logical relations among statements that are treated as unanalysed units, each either true or false. Compound statements are built from atomic propositions using the connectives negation ($\neg$), conjunction ($\wedge$), disjunction ($\vee$), implication ($\to$) and equivalence ($\leftrightarrow$). The modern algebraic treatment began with Boole in the mid-nineteenth century, and the Stoics had earlier studied inference patterns among propositions.

## Semantics and validity
A valuation assigns a truth value to each atomic proposition, and the truth value of a compound is computed from its parts by truth tables. A formula is a tautology if it is true under every valuation, satisfiable if true under at least one, and a contradiction if true under none. An argument is valid when no valuation makes all premises true and the conclusion false. Familiar valid patterns include modus ponens, from $p$ and $p\to q$ infer $q$, and modus tollens, from $\neg q$ and $p\to q$ infer $\neg p$.

## Deductive systems and metatheory
Proof can be organised in axiomatic (Hilbert-style) systems, natural deduction, sequent calculi or resolution. Classical propositional logic is sound and complete: a formula is derivable exactly when it is a tautology. It is also decidable, since a truth table settles any formula in finite time, although the table grows exponentially with the number of atoms. Cook (1971) showed that determining satisfiability is NP-complete, which makes it a central problem in computational complexity and the basis of modern SAT solvers.

Non-classical variants such as intuitionistic and many-valued logics modify the semantic or proof-theoretic assumptions, for example by rejecting the law of excluded middle.

## Relation to Active Inference
The connection is mostly through generalisation. Probability theory assigns degrees of belief to propositions, and with values restricted to 0 and 1 it reduces to classical propositional reasoning, a point emphasised by Jaynes. Bayesian networks and the factorised models used in discrete Active Inference are defined over variables that can be read as propositions, but the framework works with probability distributions and does not use logical deduction.

## Related concepts
- [[predicate_logic]] — adds quantifiers and relations
- [[mathematical_logic]] — the field that studies formal systems
- [[symbolic_logic]] — the notational tradition
- [[deductive_reasoning]] — the psychology of drawing logical inferences
- [[syllogistic_logic]] — an older, term-based logic
- [[conditional_reasoning]] — human reasoning with if-then statements

## References
- Enderton, H. B. (2001). A Mathematical Introduction to Logic (2nd ed.). Harcourt/Academic Press.
- Cook, S. A. (1971). The complexity of theorem-proving procedures. Proceedings of the Third Annual ACM Symposium on Theory of Computing, 151-158.
- Jaynes, E. T. (2003). Probability Theory: The Logic of Science. Cambridge University Press.
