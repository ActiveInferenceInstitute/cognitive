---
title: Decoherence Theory
description: The study of how interaction with an environment suppresses quantum interference between states of a system, so that the system's reduced state looks classical in a preferred basis.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - quantum_mechanics
  - open_quantum_systems
  - quantum_to_classical
aliases:
  - decoherence
semantic_relations:
  - type: foundation
    links:
      - "[[quantum_mechanics]]"
      - "[[quantum_information]]"
  - type: relates
    links:
      - "[[measurement_problem]]"
      - "[[quantum_systems]]"
      - "[[quantum_cognition]]"
      - "[[markov_blankets]]"
      - "[[statistical_physics]]"
---

# Decoherence Theory

## Overview
Decoherence is the suppression of interference that follows when a quantum system becomes entangled with its environment. The Stanford Encyclopedia of Philosophy entry by Bacciagaluppi describes the phenomenon as the suppression of interference, which arises because interference effects are only present in the larger combined system and not in the system alone. Decoherence theory covers two overlapping research areas: environmental decoherence, which studies concrete models of system-environment interaction, and the decoherent histories formalism.

## A minimal model
Let a system $S$ be in a superposition $\sum_i c_i |s_i\rangle$ and let the environment $E$ start in $|E_0\rangle$. If the interaction correlates each system state with a distinct environment state, the joint state becomes

$$|\Psi\rangle = \sum_i c_i\, |s_i\rangle \otimes |E_i\rangle .$$

Observers with access only to $S$ use the reduced density operator $\rho_S=\operatorname{Tr}_E|\Psi\rangle\langle\Psi|$, whose entries in the basis $\{|s_i\rangle\}$ are

$$(\rho_S)_{ij}=c_i c_j^{*}\,\langle E_j|E_i\rangle .$$

The diagonal terms $|c_i|^2$ are untouched. The off-diagonal terms, which carry the interference, are multiplied by the overlap $\langle E_j|E_i\rangle$ and vanish as the environment states become orthogonal. Realistic environments contain very many degrees of freedom, so the overlap typically falls quickly. The basis in which this happens is fixed by the form of the interaction. Zurek (2003) called the process by which the environment selects such a stable set of "pointer" states einselection.

## What it does and does not explain
Joos and Zeh (1985) showed how interaction with the environment makes macroscopic superpositions unobservable in practice. The reduced state is then approximately a classical mixture, but it is a mixture because the correlations with the environment have been ignored, not because one alternative has been realised. For this reason the Stanford Encyclopedia entry stresses that decoherence does not by itself solve the measurement problem: it explains why superpositions of measurement results are not observed, not why a single definite result occurs. It can contribute to a solution only together with a foundational approach, such as Everett, Bohm or collapse models (see [[measurement_problem]]). Schlosshauer (2004) reviews these points in detail.

## Relation to Active Inference
The link is indirect. The free energy principle is stated for classical stochastic systems with a Markov blanket, and an environment that is traced out plays a role loosely similar to the external states beyond the blanket. The quantum extension of Fields, Friston, Glazebrook and Levin (2022) treats system and environment as quantum systems separated by a boundary, but it is a research proposal, not an accepted part of decoherence theory. Decoherence is also invoked in arguments about whether quantum effects in warm neural tissue could matter. Tegmark (2000) estimated decoherence times for neural processes that are far too short for them to matter, which is why [[quantum_cognition]] confines itself to quantum probability as a modelling language and does not posit quantum states in neurons.

## Related concepts
- [[quantum_mechanics]] — the formalism in which the density operator is defined
- [[quantum_information]] — entanglement and entropy of the reduced state
- [[measurement_problem]] — the question decoherence leaves open
- [[quantum_systems]] — open-system dynamics and the Lindblad equation
- [[markov_blankets]] — the boundary construction in the classical theory

## References
- Bacciagaluppi, G. The Role of Decoherence in Quantum Mechanics. Stanford Encyclopedia of Philosophy (first published 3 November 2003). https://plato.stanford.edu/entries/qm-decoherence/
- Zurek, W. H. (2003). Decoherence, einselection, and the quantum origins of the classical. Reviews of Modern Physics, 75(3), 715-775. https://doi.org/10.1103/RevModPhys.75.715
- Joos, E., & Zeh, H. D. (1985). The emergence of classical properties through interaction with the environment. Zeitschrift fur Physik B, 59(2), 223-243. https://doi.org/10.1007/BF01725541
- Schlosshauer, M. (2004). Decoherence, the measurement problem, and interpretations of quantum mechanics. Reviews of Modern Physics, 76(4), 1267-1305. https://doi.org/10.1103/RevModPhys.76.1267
- Tegmark, M. (2000). Importance of quantum decoherence in brain processes. Physical Review E, 61(4), 4194-4206. https://doi.org/10.1103/PhysRevE.61.4194
- Fields, C., Friston, K., Glazebrook, J. F., & Levin, M. (2022). A free energy principle for generic quantum systems. Progress in Biophysics and Molecular Biology, 173, 36-59. https://doi.org/10.1016/j.pbiomolbio.2022.05.006
