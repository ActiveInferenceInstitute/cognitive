---
title: Autocatalysis
description: A chemical process in which a product catalyses its own formation, producing positive feedback, sigmoidal growth and, in networks, collectively self-sustaining reaction sets relevant to self-organisation and origin-of-life theory.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - chemistry
  - self_organization
  - origin_of_life
semantic_relations:
  - type: relates
    links:
      - "[[self_organization]]"
      - "[[autopoiesis]]"
      - "[[organizational_closure]]"
      - "[[feedback_loops]]"
      - "[[reaction_diffusion]]"
---

# Autocatalysis

## Overview
A reaction is autocatalytic when one of its products acts as a catalyst for the same reaction. The simplest scheme is $A + X \to 2X$, in which $X$ promotes its own production from a substrate $A$. Autocatalysis is a form of positive feedback, so it can amplify small initial amounts of a product into large ones.

## Kinetics
For the scheme above with rate constant $k$, the rate is $dx/dt = k a x$. When the substrate is large and roughly constant, growth is exponential. As the substrate is consumed, growth slows, giving a sigmoidal time course with a lag phase followed by a rapid rise. This lag-then-acceleration signature is used experimentally to identify autocatalytic steps. Classical examples include certain acid-catalysed hydrolyses in which the released acid speeds the reaction, the formose reaction, and the autocatalytic steps in oscillating reactions such as the Belousov-Zhabotinsky system. Combined with diffusion, autocatalysis can generate travelling fronts and spatial patterns (see [[reaction_diffusion]]).

## Autocatalytic sets
In network form, no single molecule needs to catalyse itself. Kauffman proposed that a sufficiently diverse set of molecules, each formed by a reaction catalysed by another member of the set, can become collectively autocatalytic. Eigen and Schuster described hypercycles, in which replicators catalyse one another cyclically. Later formal work, including the theory of reflexively autocatalytic and food-generated sets, studies when such sets arise. These ideas belong to hypotheses about the origin of life and are discussed alongside metabolism-first and replicator-first views, which remain debated.

## Relation to Active Inference and the vault's themes
Autocatalytic closure is a chemical precursor to the closure of constraints discussed under [[organizational_closure]] and [[autopoiesis]]: a system that produces the components needed to maintain itself. The Free Energy Principle describes self-maintaining systems statistically through Markov blankets, but it does not derive autocatalytic chemistry. The relation is a conceptual neighbourhood, not a derivation. The page on [[emergence_self_organization]] uses autocatalysis as an example of positive feedback driving organisation.

## Related concepts
- [[self_organization]] — autocatalytic feedback is a common driver of spontaneous order
- [[autopoiesis]] — self-producing organisation that autocatalytic sets resemble
- [[organizational_closure]] — closure of constraints in living systems
- [[metabolic_networks]] — networks in which catalytic loops occur
- [[feedback_loops]] — positive feedback as the underlying structure

## References
- Kauffman, S. A. (1986). Autocatalytic sets of proteins. Journal of Theoretical Biology, 119, 1-24.
- Kauffman, S. A. (1993). The Origins of Order: Self-Organization and Selection in Evolution. Oxford University Press.
- Eigen, M., & Schuster, P. (1979). The Hypercycle: A Principle of Natural Self-Organization. Springer.
