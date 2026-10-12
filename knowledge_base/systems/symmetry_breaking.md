---
title: Symmetry Breaking
description: "Symmetry breaking is the process by which a system governed by symmetric laws settles into a less symmetric state, central to phase transitions, pattern formation, morphogenesis and decision-making in collectives."
type: system
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - self_organization
  - phase_transitions
  - pattern_formation
semantic_relations:
  - type: relates
    links:
      - "[[phase_transitions]]"
      - "[[self_organization]]"
      - "[[emergence]]"
      - "[[dissipative_structures]]"
wikidata: Q903282
---

# Symmetry Breaking

## Overview

A system has a symmetry when its dynamics are unchanged under some transformation, such as reflection or rotation. Symmetry breaking occurs when the system's actual state does not share the symmetry of its governing laws or conditions. The distinction between explicit breaking (the law itself contains an asymmetry) and spontaneous breaking (the law is symmetric but the realised state is not) is the key one for self-organisation, and spontaneous breaking is the sense usually meant in the complex-systems literature.

## Mechanism

The standard minimal model is the Landau potential $V(x) = -\tfrac{a}{2}x^2 + \tfrac{b}{4}x^4$ with $b>0$, which is symmetric under $x \to -x$. When $a<0$ the only minimum is $x=0$ and the symmetry is respected. When $a>0$, the point $x=0$ becomes a maximum and two equivalent minima appear at $x=\pm\sqrt{a/b}$. The system must fall into one of them; which one is decided by a small fluctuation or bias. Before the transition the order parameter is zero, afterwards it takes one of several equivalent nonzero values. Because the choice amplifies a tiny fluctuation, outcomes can be unpredictable even though the dynamics are deterministic.

## Examples

- **Ferromagnets**: above the Curie temperature spins are disordered and no direction is preferred; below it they align along one of equivalent directions.
- **Convection and Turing patterns**: a uniform state becomes unstable and stripes or spots form ([[dissipative_structures]]).
- **Morphogenesis**: an initially symmetric embryo develops left-right or anterior-posterior asymmetry, with biochemical feedback amplifying small differences.
- **Collective decisions**: a group faced with two equal options commits to one, since positive feedback (recruitment) amplifies early differences; ant trail choice between two equal paths is a standard example ([[collective_behavior_ants]]).
- **Particle physics**: spontaneous breaking in the Higgs mechanism gives masses to some gauge bosons.

## Role in Active Inference

The free-energy principle links to the topic through the physics of non-equilibrium systems and through models of morphogenesis in which cells with identical generative models acquire different roles by differing in their local observations, which breaks symmetry across the tissue. A multimodal free-energy landscape with equivalent minima gives the decision-making analogue: the agent must commit to one policy, with noise or a small prior asymmetry selecting it. In expected free energy, tied options are resolved by the stochastic selection of policies, for example via a softmax over policy values. These are applications of the general mathematics of bifurcations; the principle does not add a distinct account of symmetry breaking, and the equivalence of a given symmetry-breaking system to an inference process must be argued for each case.

## Related concepts

- [[phase_transitions]] — transitions where order parameters appear
- [[self_organization]] — order arising without external design
- [[emergence]] — new collective-level structure
- [[dissipative_structures]] — patterned states far from equilibrium
- [[critical_phenomena]] — behaviour near symmetry-breaking transitions
- [[developmental_systems]] — symmetry breaking in embryonic development
