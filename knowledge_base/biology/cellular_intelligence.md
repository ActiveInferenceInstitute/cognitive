---
title: Cellular Intelligence
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-08-02
tags:
  - biology
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - [[knowledge_base/biology/cell_biology]]
      - [[knowledge_base/cognitive/active_inference]]
---

# Cellular Intelligence

## Overview

Cellular intelligence refers to the capacity of individual cells to sense
their environment, process information, and act adaptively without a
centralized controller. Signal transduction, chemotaxis, and
morphogenetic computation are canonical examples.

## Signal Transduction

Cells convert extracellular signals into intracellular responses through
receptor binding, second-messenger cascades, and gene-regulatory
feedback. Under an active-inference reading, the cell maintains a
generative model of its environment and acts to resolve prediction
error, with the cell membrane as a Markov blanket separating internal
states from external causes.

## Chemotaxis

Chemotaxis is the directed movement of a cell along a chemical gradient.
Bacteria alternate between run and tumble phases; the run length is
biased upward when the gradient is favorable. This biased random walk can
be framed as active inference over hidden environmental states, where the
cell's belief about the gradient drives its policy selection.

## Morphogenetic Computation

Morphogenesis — the development of form — involves cells exchanging
signals to coordinate differentiation and pattern formation. Reaction-
diffusion dynamics, lateral inhibition, and Turing patterns are the
classical mechanisms; active inference offers a variational account in
which developing tissues minimize free energy over their configuration.

## Related Resources

- [[knowledge_base/biology/cell_biology]]
- [[knowledge_base/cognitive/active_inference]]
