---
title: Axon Guidance
description: The molecular cues, including netrins, slits, semaphorins and ephrins, that steer growing axons to their targets, and how gradient following by growth cones relates to inference-based accounts of development.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neurodevelopment
  - signalling
  - neurons
semantic_relations:
  - type: relates
    links:
      - "[[axon_growth]]"
      - "[[neural_development]]"
      - "[[morphogenesis]]"
      - "[[pattern_formation]]"
      - "[[free_energy_principle]]"
---

# Axon Guidance

## Overview
Axon guidance is the process by which a growing axon finds its way through the embryonic nervous system to its correct targets. The growth cone at the axon tip reads cues in its environment and turns toward, or away from, them. Roger Sperry's chemoaffinity hypothesis, based on regeneration experiments in the visual system, proposed that matching molecular labels on axons and target tissue specify connections. This idea has been largely supported, though it is complemented by activity-dependent refinement.

## Guidance cues
Cues are classified by whether they act at a distance (diffusible) or on contact, and whether they attract or repel. Four large families are central:
- **Netrins** act mostly as attractants through DCC receptors and can repel through other receptors.
- **Slits** repel through Robo receptors.
- **Semaphorins** act mainly as repellents through plexin and neuropilin receptors.
- **Ephrins and Eph receptors** act on contact and form gradients that pattern topographic maps such as the retinotectal projection.

A well-studied case is the midline crossing of commissural axons. They are first attracted by midline netrin, and once they have crossed, their responsiveness to Slit changes so that they are repelled from the midline and do not recross. Cues are often combined, so that a growth cone integrates attractive and repulsive signals, and its sensitivity can adapt as it moves.

## Relation to Active Inference
Growth cones follow chemical gradients, which resembles gradient descent on a quantity, and the Free Energy Principle has been applied to simple chemotaxing agents and to morphogenesis. However, axon guidance has not been developed as a specific Active Inference model, and the link to [[free_energy_principle]] is an analogy: a self-organising system tuned by evolution to reach a viable configuration. The molecular account stands independently of any such reading.

## Related concepts
- [[axon_growth]] — the cytoskeletal machinery that guidance cues regulate
- [[neural_development]] — the developmental context for wiring
- [[cell_migration]] — neurons and growth cones use shared guidance molecules
- [[morphogenesis]] — pattern formation by gradients in embryos
- [[critical_periods]] — later stages when experience refines connections

## References
- Sperry, R. W. (1963). Chemoaffinity in the orderly growth of nerve fiber patterns and connections. Proceedings of the National Academy of Sciences, 50, 703-710.
- Tessier-Lavigne, M., & Goodman, C. S. (1996). The molecular biology of axon guidance. Science, 274, 1123-1133.
- Dickson, B. J. (2002). Molecular mechanisms of axon guidance. Science, 298, 1959-1964.
