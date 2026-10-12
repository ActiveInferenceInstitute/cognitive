---
title: Scientific Realism
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - philosophy_of_science
  - realism
  - free_energy_principle
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[knowledge_base/cognitive/free_energy_principle]]"
      - "[[instrumentalism]]"
wikidata: Q431354
---

# Scientific Realism

## Overview
Scientific realism is the view that our best mature scientific theories are approximately true descriptions of a mind-independent world, including its unobservable parts. It is usually split into a metaphysical claim (the world has a structure independent of our theories), a semantic claim (theoretical statements are read literally and are truth-apt), and an epistemic claim (we have good reason to believe the successful ones).

## Core claims
**No-miracles argument.** Putnam's formulation: if theories were not at least approximately true, the predictive success of science would be a miracle. It is an inference to the best explanation, so critics note that it is circular when used against anti-realists who reject such inferences.

**Pessimistic meta-induction.** Laudan listed once-successful theories that were later discarded (caloric, the luminiferous ether), and argued that current success does not entail truth.

**Underdetermination.** Empirically equivalent theories may disagree about unobservables, so evidence alone cannot select one.

**Selective retreats.** Entity realism (Hacking) trusts entities we can manipulate to produce effects elsewhere. Structural realism (Worrall) holds that what survives theory change is mathematical structure, not ontology. Both are attempts to keep the no-miracles intuition while conceding the meta-induction.

## Role in Active Inference
The realism question arises twice. First, within a model: is the hidden-state space of a generative model a description of the world, or only whatever partition makes predictions work? A model with a column-stochastic likelihood `A` and transitions `B` commits only to the structure of the mapping between latent states and outcomes, which fits structural realism better than entity realism. Second, about the framework: whether the free-energy principle describes real systems (Markov blankets as physical boundaries) or is a modelling lens whose blankets are drawn by the observer is an open dispute in the literature, with realist and instrumentalist readings both defended. The structural reading is the most defensible for the repo's own code, which validates model structure through tests rather than through metaphysical claims.

This page makes no claim that the principle itself is true of any given organism; that is an empirical and interpretive question, discussed under [[instrumentalism]] and [[dark_room_problem]].

## Related concepts
- [[instrumentalism]] — the main rival: theories as predictive tools
- [[pragmatism]] — truth as what survives inquiry, a middle path
- [[operationalism]] — meaning tied to measurement operations, historically opposed to realism about unobservables
- [[epistemology]] — the FEP read as a formal epistemology
- [[markov_blanket]] — the construct whose reality status is most contested
- [[philosophy_of_mind]] — realism about representations in generative models

## References
- Putnam, H. (1975). *Mathematics, Matter and Method*.
- Laudan, L. (1981). A Confutation of Convergent Realism. *Philosophy of Science*.
- van Fraassen, B. (1980). *The Scientific Image*.
- Hacking, I. (1983). *Representing and Intervening*.
- Worrall, J. (1989). Structural Realism: The Best of Both Worlds? *Dialectica*.
- Andrews, M. (2021). The Math is not the Territory: Navigating the Free Energy Principle. *Biology & Philosophy*.
