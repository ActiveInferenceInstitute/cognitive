---
title: Landauer Principle
description: The thermodynamic lower bound on the heat dissipated when one bit of information is erased, its derivation, experimental tests, and its status as a bridge between information and physics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - thermodynamics
  - information_theory
  - computation
semantic_relations:
  - type: relates
    links:
      - "[[information_thermodynamics]]"
      - "[[maxwell_demon]]"
      - "[[entropy]]"
      - "[[thermodynamics]]"
      - "[[free_energy_principle]]"
---

# Landauer Principle

## Overview
Landauer's principle states that erasing one bit of information in a system coupled to a heat reservoir at temperature $T$ must dissipate at least $k_B T \ln 2$ of heat into that reservoir, where $k_B$ is Boltzmann's constant. At 300 K this is roughly $3\times10^{-21}$ J. The bound applies to logically irreversible operations, in which several distinct input states are mapped to one output state, so that information about the input is lost.

## Argument
Resetting a memory cell that may be in either of two states to a fixed standard state halves the number of accessible logical states. The entropy of the memory falls by $k_B\ln 2$, and the second law requires that the total entropy of memory plus environment not decrease. The environment must therefore gain at least $k_B\ln 2$ of entropy, which at fixed temperature corresponds to heat $Q \ge k_B T\ln 2$. Operations that are logically reversible, such as a controlled swap, have no such lower limit, which is the basis of Bennett's analysis of reversible computation. The bound is also central to the standard resolution of the [[maxwell_demon]] paradox: the demon's memory must eventually be erased, and that cost offsets the work extracted.

## Experimental status
The bound has been tested in single-particle experiments in which a colloidal bead in a modulated double-well potential serves as a one-bit memory. The measured heat approaches $k_B T\ln 2$ in the limit of slow erasure and exceeds it for finite-time protocols. Practical processors dissipate many orders of magnitude more heat per operation than this limit.

## Relation to Active Inference
The principle supplies the physical link between information and energy that many arguments in this vault take for granted. Some authors in the free energy literature invoke it to argue that minimising variational free energy is related to thermodynamic efficiency, since an agent that keeps a compact, accurate model discards less information per unit of inference. This is an active but contested line of work and relies on mapping informational quantities onto thermodynamic ones, which holds exactly only under specific assumptions. For most applications of Active Inference the connection is conceptual and does not enter calculations.

## Related concepts
- [[information_thermodynamics]] — the wider framework in which the bound sits
- [[maxwell_demon]] — the thought experiment whose resolution relies on erasure cost
- [[work_extraction]] — how much work information can in principle yield
- [[entropy]] — the quantity whose reduction on erasure sets the bound
- [[free_energy_principle]] — the framework that draws on information-thermodynamic arguments

## References
- Landauer, R. (1961). Irreversibility and heat generation in the computing process. IBM Journal of Research and Development, 5, 183-191.
- Bennett, C. H. (1982). The thermodynamics of computation: a review. International Journal of Theoretical Physics, 21, 905-940.
- Bérut, A., Arakelyan, A., Petrosyan, A., Ciliberto, S., Dillenschneider, R., & Lutz, E. (2012). Experimental verification of Landauer's principle linking information and thermodynamics. Nature, 483, 187-189.
