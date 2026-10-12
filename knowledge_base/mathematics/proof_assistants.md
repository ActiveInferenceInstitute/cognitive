---
title: Proof Assistants
description: Software systems in which mathematical statements and proofs are written in a formal language and checked by a small trusted kernel, including Coq, Isabelle, Lean and Agda, and their logical foundations.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - formal_verification
  - type_theory
  - mathematical_logic
semantic_relations:
  - type: relates
    links:
      - "[[type_theory]]"
      - "[[proof_theory]]"
      - "[[mathematical_logic]]"
      - "[[set_theory]]"
      - "[[ai_for_mathematics_cognitive_perspective]]"
---

# Proof Assistants

## Overview
A proof assistant, also called an interactive theorem prover, is a program for developing machine-checked proofs. The user states definitions and theorems in a formal language and supplies proofs, either directly as terms or through tactics, which are commands that build a proof step by step. A kernel then checks that the result is a valid derivation in the underlying logic. Because only the kernel must be trusted, a large body of mathematics can be verified while relying on a small amount of code, a design principle sometimes called the de Bruijn criterion.

## Foundations and systems
Most modern systems rest on one of two foundations. Type-theoretic systems such as Coq (now developed as Rocq), Lean and Agda are based on dependent type theory, in which propositions are types and proofs are programs inhabiting them, an idea called the Curry-Howard correspondence. Isabelle/HOL is based on classical higher-order logic with a simple type system (Nipkow et al., 2002). Set-theoretic systems such as Mizar are built on first-order logic with set theory. Foundations differ in whether they are constructive by default, how they treat equality and quotients, and whether they provide a hierarchy of universes.

Notable formalisations include the four colour theorem in Coq, the Kepler conjecture in a collaboration using HOL Light and Isabelle, and the large community library mathlib for Lean. Proof assistants are also used to verify software and hardware, for example a compiler and an operating-system kernel.

Automation in these systems ranges from simple decision procedures and rewriting to calls to external solvers and, more recently, machine-learned suggestions. Whatever generates a proof, the kernel's check provides the guarantee, so learned components do not need to be trusted.

## Relation to Active Inference
Proof assistants have no theoretical relationship to Active Inference. They are relevant to the vault's themes in two practical ways. Formalising derivations, such as properties of variational bounds or constructions in category theory, makes their assumptions explicit. And discussion of how people and machines find proofs belongs to the study of mathematical cognition. Neither makes proof construction an application of the free energy principle.

## Related concepts
- [[type_theory]] — the logical basis of several proof assistants
- [[proof_theory]] — the formal study of derivations
- [[mathematical_logic]] — the logics that kernels implement
- [[set_theory]] — the alternative foundation used by some systems
- [[category_theory]] — a subject increasingly formalised in these systems
- [[ai_for_mathematics_cognitive_perspective]] — machine assistance in mathematics

## References
- Nipkow, T., Paulson, L. C., & Wenzel, M. (2002). Isabelle/HOL: A Proof Assistant for Higher-Order Logic. Lecture Notes in Computer Science 2283. Springer.
- Bertot, Y., & Castéran, P. (2004). Interactive Theorem Proving and Program Development: Coq'Art. Springer.
- de Moura, L., Kong, S., Avigad, J., van Doorn, F., & von Raumer, J. (2015). The Lean theorem prover (system description). Lecture Notes in Computer Science 9195, 378-388.
