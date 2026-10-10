---
title: Cellular Automata
description: "Cellular automata are discrete lattices of locally interacting cells with synchronous update rules, used to study emergence, self-organisation, computation and morphogenesis, with limited direct use in Active Inference."
type: system
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - complexity
  - emergence
  - computation
semantic_relations:
  - type: relates
    links:
      - "[[emergence]]"
      - "[[artificial_life]]"
      - "[[complex_systems]]"
      - "[[interaction_rules]]"
---

# Cellular Automata

## Overview

A cellular automaton (CA) is a discrete dynamical system defined on a regular grid of cells, each in one of finitely many states. At each time step all cells update simultaneously, and each cell's new state depends only on its own state and those of its neighbours, according to one fixed rule. Despite this simplicity, CAs show ordered, chaotic and computationally rich behaviour, which makes them standard models of [[emergence]].

## Definition

A CA consists of four components: a lattice (for example a line or a two-dimensional grid), a finite state set, a neighbourhood (for example the nearest cells on each side), and a local update rule applied uniformly. In the one-dimensional, two-state, nearest-neighbour case, each cell sees a neighbourhood of three cells, so there are $2^3 = 8$ neighbourhood patterns and $2^8 = 256$ possible rules. These are the elementary CAs indexed by Wolfram.

## Behaviour

- **Conway's Game of Life** (two-dimensional, birth with three live neighbours, survival with two or three) produces gliders, oscillators and structures that can implement logic gates; it is Turing complete.
- **Rule 110**, an elementary CA, has been shown to be Turing complete, so simple local rules can support universal computation.
- **Wolfram's classes** group rules by long-run behaviour: homogeneous, periodic, chaotic and complex. The classification is empirical and not a formal partition.
- **Reversibility and conservation** can be built into rules, which allows CAs to model physical systems such as lattice gases.
- **Neural and continuous variants** (for example neural cellular automata with learned update rules) have been used for growth and regeneration of patterns, a link to developmental biology.

## Uses

CAs model reaction-diffusion patterns, traffic flow, epidemic spread and excitable media, and in biology they illustrate how local cell-cell signalling can give rise to tissue-level patterns. They show that global order does not require a central controller, the same point made in studies of ant colonies and swarms.

## Role in Active Inference

The link is indirect. The free-energy principle has been illustrated by simulations of coupled lattices and of morphogenesis, where each cell holds a generative model of its local signalling environment and the collective converges on a target form; those are Active Inference simulations rather than CAs in the strict sense, since the updates are inference dynamics instead of lookup rules. A CA can be viewed as the limiting case in which an agent's policy is a fixed deterministic mapping from local observations, with no learning, no uncertainty and no goals. Using CAs to test claims about Markov blankets or self-organisation is possible but requires defining the blanket statistically, which has been done only in specific toy settings.

## Related concepts

- [[emergence]] — global patterns from local rules
- [[artificial_life]] — CAs as substrates for life-like dynamics
- [[developmental_systems]] — pattern formation in tissues
- [[complex_systems]] — the field in which CAs are a canonical model
- [[interaction_rules]] — local rules as the basis of collective behaviour
- [[symmetry_breaking]] — pattern formation from uniform initial conditions
