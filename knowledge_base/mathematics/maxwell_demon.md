---
title: Maxwell's Demon
description: The thought experiment of a being that sorts molecules to apparently violate the second law, its resolution through measurement and memory erasure, and experimental realisations and links to information thermodynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - thermodynamics
  - information_theory
  - thought_experiment
semantic_relations:
  - type: relates
    links:
      - "[[information_thermodynamics]]"
      - "[[landauer_principle]]"
      - "[[thermodynamics]]"
      - "[[entropy]]"
      - "[[work_extraction]]"
---

# Maxwell's Demon

## Overview
Maxwell's demon is a thought experiment proposed by James Clerk Maxwell in 1867 to probe the statistical nature of the second law of thermodynamics. A small being controls a door between two chambers of gas at the same temperature. By opening the door for fast molecules moving one way and slow molecules the other, it appears to create a temperature difference without doing work, which would decrease entropy and allow work to be extracted from a single heat bath.

## Resolution
Early analysis focused on the cost of measurement. The Szilard engine, a one-molecule version proposed in 1929, showed that learning which side of a box the molecule occupies, one bit of information, enables extraction of $k_BT\ln 2$ of work. Brillouin and others argued that the measurement itself must dissipate energy. Later work by Landauer and Bennett shifted the emphasis: measurement can in principle be done reversibly, but the demon's memory is finite, and resetting it for the next cycle erases a bit and dissipates at least $k_BT\ln 2$ of heat (see [[landauer_principle]]). This cost compensates for the extracted work, so the second law holds once the demon and its memory are included in the accounting. Modern treatments in stochastic thermodynamics make this quantitative through generalised second laws that include mutual information.

## Experiments
Small systems have been used to realise demons. In one experiment, a colloidal particle on a spiral-staircase-like potential was monitored, and a feedback protocol that raised the potential behind the particle's upward fluctuations converted measured information into free energy, in agreement with a generalised Jarzynski equality. Other realisations used single-electron devices and optical systems.

## Relation to Active Inference
The demon is a conceptual ancestor of ideas that treat an organism as an information-processing system that measures its environment and acts to maintain a low-entropy state. The free energy principle, with sensory and active states on a Markov blanket, is sometimes compared to a demon: it senses, updates a model and acts. This is an analogy. The thermodynamic costs discussed here concern physical memory and measurement devices, while variational free energy is an information-theoretic quantity, and whether biological inference approaches physical bounds is an open question.

## Related concepts
- [[information_thermodynamics]] — the modern framework that formalises the resolution
- [[landauer_principle]] — the erasure cost that rescues the second law
- [[thermodynamic_computing]] — limits on the energy cost of computation
- [[entropy]] — the quantity the demon appears to decrease
- [[markov_blanket]] — the boundary compared to the demon's door
- [[work_extraction]] — converting information into work

## References
- Leff, H. S., & Rex, A. F. (2003). Maxwell's Demon 2: Entropy, Classical and Quantum Information, Computing. Institute of Physics Publishing.
- Bennett, C. H. (1982). The thermodynamics of computation: a review. International Journal of Theoretical Physics, 21, 905-940.
- Toyabe, S., et al. (2010). Experimental demonstration of information-to-energy conversion and validation of the generalized Jarzynski equality. Nature Physics, 6, 988-992.
