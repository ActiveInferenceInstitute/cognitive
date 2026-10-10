---
title: Tensegrity Systems
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - research
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/systems/tensegrity]]"
      - "[[knowledge_base/free_energy_principle/README]]"
      - "[[knowledge_base/mathematics/tensegrity_information_geometry]]"
      - "[[knowledge_base/systems/synergetics]]"
      - "[[knowledge_base/research/design_science]]"
      - "[[knowledge_base/research/dymaxion_systems]]"
      - "[[knowledge_base/mathematics/markov_blankets]]"
---

# Tensegrity Systems

## Overview

A tensegrity (tensional integrity) structure is a pre-stressed assembly in which a set of isolated compression members, the struts, is held in a stable shape by a continuous network of tension members, the cables. Stability comes from the balance of forces across the whole structure and not from any member carrying load alone. The name was coined by Buckminster Fuller; the first sculptures are credited to Kenneth Snelson, who made them in 1948 after studying with Fuller at Black Mountain College. Fuller patented the concept in 1962 and Snelson took out his own patent a few years later; the question of priority has been disputed. Earlier constructivist work by Karl Ioganson in the 1920s is often cited as a precursor.

## Mechanics

A tensegrity is rigid only when it is in a state of self-stress, so the cables must carry tension before any external load is applied. When it is loaded, the force spreads through the network and the structure deforms as a whole, often with large elastic deformations and a high ratio of strength to mass. The equilibrium shape is a stationary point of the elastic potential energy, and form-finding means searching for geometries in which every cable is in tension and every strut in compression at equilibrium. Related engineering applications include lightweight roofs and bridges, deployable structures and tensegrity robots, such as the NASA SUPERball concept, which uses the structure's compliance to absorb impact.

## Tensegrity in biology

Donald Ingber proposed in the 1990s that cells maintain shape through a pre-stressed cytoskeleton in which microtubules act as struts and actin filaments act as cables (Ingber, 1993; 1998). Part of this is well supported: cells do exert and sense mechanical forces, and mechanical signals alter gene expression and cell fate (mechanotransduction). The stronger claim that the whole cytoskeleton behaves as a tensegrity is debated, with alternative models of cell mechanics available. The term "biotensegrity" has also been extended to whole-body anatomy and fascia, where the evidence is much weaker and the claims in popular and therapeutic literature should be treated cautiously.

## Connection to Active Inference

The connection is an analogy, with formal development in this repository that should be read as exploratory. Tensegrity illustrates in a concrete physical system how global stability and adaptivity can arise from local opposing forces that settle into a minimum of an energy function. The free energy principle describes how a persistent system's internal states relax toward a non-equilibrium steady state, and this settling is often compared to a mechanical system finding its equilibrium. The word "free energy" in the principle denotes a variational bound on surprise, not the thermodynamic or elastic energy of the structure, so the likeness is structural and no more. A tensegrity can also be pictured as a body with a boundary, where internal tension differs from external load, which is the intuition behind Markov blankets. The repository's [[knowledge_base/mathematics/tensegrity_information_geometry]] page attempts to give this a geometric statement. It has not been tested against data.

## In this repository

See [[knowledge_base/systems/tensegrity]] for the systems-level treatment and [[knowledge_base/systems/synergetics]] for Fuller's geometry. Wider context is in [[knowledge_base/research/design_science]] and [[knowledge_base/research/dymaxion_systems]]. Free energy background is in [[knowledge_base/mathematics/markov_blankets]]. A study path is [[active_inference_fuller_learning_path]].

## References

- Ingber, D. E. (1993). Cellular tensegrity: defining new rules of biological design that govern the cytoskeleton. *Journal of Cell Science*, 104, 613-627.
- Ingber, D. E. (1998). The architecture of life. *Scientific American*, 278(1), 48-57.
- Fuller, R. B., with Applewhite, E. J. (1975). *Synergetics*. Macmillan.

## Related Resources

- [[knowledge_base/systems/tensegrity]]
- [[knowledge_base/free_energy_principle/README]]
