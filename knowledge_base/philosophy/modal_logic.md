---
title: Modal Logic
description: Logics of necessity and possibility, their Kripke semantics and standard systems, and the way modal and counterfactual reasoning relates loosely to planning in generative models.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - logic
  - modality
  - formal_systems
semantic_relations:
  - type: relates
    links:
      - "[[propositional_logic]]"
      - "[[predicate_logic]]"
      - "[[temporal_logic]]"
      - "[[symbolic_logic]]"
      - "[[metaphysics]]"
wikidata: Q210841
---

# Modal Logic

## Overview
Modal logic extends classical logic with operators for modes of truth. The standard operators are $\Box$, read "necessarily", and $\Diamond$, read "possibly", usually related by $\Diamond\varphi \leftrightarrow \neg\Box\neg\varphi$. The subject began with C. I. Lewis's work on strict implication in the early twentieth century and was transformed by relational semantics in the late 1950s and 1960s.

## Kripke semantics
A Kripke model consists of a set $W$ of worlds, a binary accessibility relation $R$ on $W$, and a valuation assigning truth values to atomic sentences at each world. The sentence $\Box\varphi$ is true at a world $w$ when $\varphi$ is true at every world accessible from $w$. Different constraints on $R$ validate different axioms:

- reflexive $R$ validates $\Box\varphi\to\varphi$ (system T);
- reflexive and transitive $R$ gives S4;
- an equivalence relation gives S5, often used for metaphysical necessity.

The minimal normal system K is sound and complete for all frames. Many modal logics are decidable, which accounts for their use in computer science.

## Interpretations
The same formal apparatus has been read in several ways: as alethic modality (necessity and possibility), epistemic logic (knowledge and belief), deontic logic (obligation and permission), and temporal logic (see [[temporal_logic]]). Philosophers use possible-worlds semantics for counterfactuals and for the analysis of properties and essence, though whether possible worlds are real is disputed and is a question for [[metaphysics]], not for the logic.

## Relation to Active Inference
The link is an analogy, not a derivation. Planning in Active Inference evaluates counterfactual futures, since expected free energy scores policies the agent has not executed, and this resembles reasoning about what would hold at accessible worlds. Epistemic logic also provides a formal vocabulary for multi-agent models in which agents represent each other's beliefs. The framework itself uses probabilistic rather than modal formalism.

## Related concepts
- [[propositional_logic]] — the base system that modal operators extend
- [[predicate_logic]] — quantified modal logic combines the two
- [[temporal_logic]] — modal logic with operators over time
- [[symbolic_logic]] — the notational tradition
- [[metaphysics]] — where the status of possible worlds is debated
- [[theory_of_mind]] — reasoning about others' beliefs, formalised in epistemic logic

## References
- Blackburn, P., de Rijke, M., & Venema, Y. (2001). Modal Logic. Cambridge University Press.
- Hughes, G. E., & Cresswell, M. J. (1996). A New Introduction to Modal Logic. Routledge.
- Kripke, S. A. (1963). Semantical considerations on modal logic. Acta Philosophica Fennica, 16, 83-94.
