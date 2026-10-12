---
title: Instrumentalism
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - philosophy_of_science
  - anti_realism
  - free_energy_principle
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[knowledge_base/cognitive/free_energy_principle]]"
      - "[[scientific_realism]]"
wikidata: Q856063
---

# Instrumentalism

## Overview
Instrumentalism holds that scientific theories are tools for organizing observations and making predictions, and that their value is exhausted by empirical adequacy and usefulness rather than by literal truth about unobservable entities. Statements about electrons or latent variables are shorthand inference licences, not descriptions to be believed.

## Core claims
**Historical sources.** Duhem and Mach argued that physical theory economizes thought and saves phenomena without revealing underlying reality. Dewey's pragmatist instrumentalism extended this: concepts are plans of action judged by whether they resolve problematic situations.

**Distinctions that matter.**
- Instrumentalism is not van Fraassen's constructive empiricism. The latter reads theories literally and withholds belief beyond empirical adequacy; classical instrumentalism denies that theoretical sentences are truth-apt at all.
- It is not operationalism, which is a theory of meaning (concepts are definable by operations); instrumentalism is a theory of the epistemic status of theories.

**Standard objections.** The no-miracles argument says a mere tool should not keep succeeding in novel predictions. Popper objected that instrumentalism ignores the testing of theories as attempts at refutation. Instrumentalists answer that adequacy across many predictions is all the success there is to explain.

## Role in Active Inference
Instrumentalism is a live reading of the free-energy principle. Friston and collaborators have compared its status to variational principles such as least action: a mathematical principle from which process theories are derived, not itself a hypothesis to refute. On that reading, applying it to an organism is a modelling choice, "as if" the system minimized free energy, and a specific generative model is accepted or rejected by Bayesian model comparison on its evidence. In this repository, `DiscreteGenerativeModel` is justified in precisely this way: its matrices `A`, `B`, `C`, `D`, `E` are validated by stochasticity constraints and by tests of identities such as the risk-plus-ambiguity decomposition, not by claims that brains contain those arrays.

The strength of the fit is moderate. The "as if" reading is straightforward for modelling; it sits less easily with claims that the brain implements specific message passing. Where authors make such implementation claims, they are making a realist commitment (see [[scientific_realism]]).

## Related concepts
- [[scientific_realism]] — the contrasting stance on theoretical entities
- [[pragmatism]] — Dewey's instrumentalism is a pragmatist doctrine
- [[operationalism]] — related empiricist family, different target
- [[logical_positivism]] — historical home of the observational/theoretical distinction
- [[dark_room_problem]] — an objection whose force depends on how the principle is read
- [[model_comparison]] — the practice that stands in for truth-assessment

## References
- Duhem, P. (1906). *La théorie physique: son objet, sa structure*.
- van Fraassen, B. (1980). *The Scientific Image*.
- Friston, K. (2019). A free energy principle for a particular physics. arXiv.
