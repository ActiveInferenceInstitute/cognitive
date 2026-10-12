---
title: Temporal Logic
description: Formal logics for reasoning about propositions whose truth changes over time, from Prior's tense logic to linear and branching time systems used in program verification, with links to planning.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - logic
  - time
  - formal_verification
semantic_relations:
  - type: relates
    links:
      - "[[modal_logic]]"
      - "[[propositional_logic]]"
      - "[[philosophy_of_time]]"
      - "[[temporal_models]]"
      - "[[planning_as_inference]]"
wikidata: Q781833
---

# Temporal Logic

## Overview
Temporal logic is a family of logics designed for statements whose truth varies with time. It is a form of modal logic in which the possible worlds are moments or states of a process and the accessibility relation orders them in time (see [[modal_logic]]). A. N. Prior introduced tense logic in the 1950s and developed it in Past, Present and Future (1967), with operators for what has been, will be, and is always the case. Two later strands, in philosophy and computer science, have developed the subject.

## Linear temporal logic
Pnueli (1977) proposed temporal logic as a way of specifying and reasoning about the behaviour of programs. In linear temporal logic (LTL), formulas are evaluated along a single infinite sequence of states, with operators including $\mathbf{X}\varphi$ (next), $\mathbf{F}\varphi$ (eventually), $\mathbf{G}\varphi$ (always) and $\varphi\,\mathbf{U}\,\psi$ (until). A safety property such as $\mathbf{G}\,\neg\mathrm{error}$ says that something bad never happens, and a liveness property such as $\mathbf{G}(\mathrm{request}\to\mathbf{F}\,\mathrm{grant})$ says that something good eventually happens.

## Branching time and model checking
In computation tree logic (CTL) formulas are interpreted over trees of possible futures, with path quantifiers such as "for all paths" and "there exists a path" combined with the temporal operators. Model checking algorithms decide whether a finite-state system satisfies a temporal formula, a technique used in hardware and protocol verification (Baier and Katoen, 2008). Temporal logics differ in expressiveness and in the computational complexity of satisfiability and model checking.

## Philosophical use
Temporal logic provides formal tools for debates about the nature of time, including whether the future is open or determinate and how tensed statements should be analysed (see [[philosophy_of_time]]). The choice of axioms, such as linear against branching structure, encodes metaphysical commitments.

## Relation to Active Inference
The link is a loose parallel. Active Inference plans over sequences of future states and scores policies by expected free energy, whereas temporal logic specifies properties of trajectories. In reinforcement learning, temporal logic is increasingly used to state complex goals for agents. Preferences in discrete Active Inference are usually specified over outcomes at each time step, so richer temporal goals would need an extension, and this has been explored only in a limited way (see [[planning_as_inference]], [[temporal_models]]).

## Related concepts
- [[modal_logic]] — the parent family of logics
- [[propositional_logic]] — the base on which temporal operators are added
- [[philosophy_of_time]] — metaphysical questions the logic is used to formalise
- [[temporal_models]] — temporal structure in cognitive models
- [[planning_as_inference]] — planning over future trajectories

## References
- Prior, A. N. (1967). Past, Present and Future. Clarendon Press.
- Pnueli, A. (1977). The temporal logic of programs. Proceedings of the 18th Annual Symposium on Foundations of Computer Science, 46-57.
- Baier, C., & Katoen, J.-P. (2008). Principles of Model Checking. MIT Press.
