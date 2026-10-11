---
title: Molecular Motors
description: Proteins and protein complexes that convert chemical energy, usually from ATP hydrolysis or ion gradients, into directed movement and force inside cells, with their main classes and physical principles.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - cell_biology
  - biophysics
  - nonequilibrium
semantic_relations:
  - type: relates
    links:
      - "[[non_equilibrium_thermodynamics]]"
      - "[[thermodynamics]]"
      - "[[free_energy_principle]]"
      - "[[cell_biology]]"
      - "[[statistical_physics]]"
---

# Molecular Motors

## Overview
Molecular motors are macromolecular machines that transduce chemical free energy into mechanical work. They drive muscle contraction, transport cargo along intracellular tracks, separate chromosomes during division, rotate bacterial flagella, copy and unwind DNA, and synthesise ATP. Because they operate at the scale of nanometres in a fluid environment, thermal fluctuations are as large as the forces they generate, and their behaviour is analysed with stochastic physics rather than with the mechanics of macroscopic engines.

## Main classes
- **Cytoskeletal motors.** Kinesins and dyneins walk along microtubules, and myosins move along actin filaments. Conventional kinesin advances in discrete steps of about 8 nm, coupled to the binding and hydrolysis of ATP, and moves processively along a single filament.
- **Rotary motors.** The F1 portion of ATP synthase rotates in steps as it hydrolyses ATP, and in cells the reverse process, driven by a proton gradient, synthesises ATP. The bacterial flagellar motor is powered by ion flow across the membrane.
- **Nucleic acid motors.** Polymerases, helicases and packaging motors translocate along DNA or RNA while performing a chemical task.

## Physical principles
Motors work far from thermodynamic equilibrium. Each cycle couples a chemical transition, such as ATP binding and hydrolysis, to a conformational change that biases a thermally driven diffusion in one direction, an idea formalised in Brownian ratchet and power-stroke models. At low Reynolds number, inertia is negligible, so motion stops essentially at once when the driving force stops. Single-molecule techniques including optical tweezers and fluorescence microscopy have measured step sizes, stall forces and dwell times, and they show considerable stochasticity from one step to the next.

## Relation to Active Inference
The relationship is physical grounding and analogy. The free energy principle assumes that living systems maintain themselves in nonequilibrium steady states by exchanging energy and matter with their surroundings, and motors are standard examples of the chemical-to-mechanical coupling that sustains such states (see [[non_equilibrium_thermodynamics]]). The variational free energy used in Active Inference is an information-theoretic quantity, distinct from the thermodynamic free energy that governs motor cycles, and no inference is attributed to the motor proteins themselves.

## Related concepts
- [[non_equilibrium_thermodynamics]] — framework for systems driven away from equilibrium
- [[thermodynamics]] — energy conversion and efficiency limits
- [[statistical_physics]] — fluctuation theory relevant to nanoscale machines
- [[cell_biology]] — the cellular processes motors carry out
- [[free_energy_principle]] — a theory of how living systems resist dispersal

## References
- Howard, J. (2001). Mechanics of Motor Proteins and the Cytoskeleton. Sunderland, MA: Sinauer Associates.
- Astumian, R. D. (1997). Thermodynamics and kinetics of a Brownian motor. Science, 276, 917-922.
- Vale, R. D. (2003). The molecular motor toolbox for intracellular transport. Cell, 112, 467-480.
