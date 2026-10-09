---
title: Biophysics
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - biophysics
  - statistical_physics
  - thermodynamics
  - membranes
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/non_equilibrium_thermodynamics]]"
      - "[[knowledge_base/mathematics/statistical_physics]]"
      - "[[knowledge_base/biology/biochemistry]]"
      - "[[knowledge_base/biology/neuroscience]]"
---

# Biophysics

## Overview
Biophysics uses the concepts and methods of physics to explain biological structure and function, at scales from single molecules to whole organisms. It covers topics such as protein folding, molecular motors, membrane transport, bioelectricity, fluid mechanics of swimming and flight, and the thermodynamics of living systems.

## Mechanism
At the molecular scale, thermal fluctuations are comparable to the energies involved in biological reactions, so processes must be described statistically. Protein folding is analyzed through energy landscapes, and single-molecule techniques such as optical tweezers and fluorescence measure forces and steps of motors like kinesin and myosin. In membrane biophysics, ion gradients and selective channels produce transmembrane potentials; the Nernst equation gives the equilibrium potential of one ion, the Goldman-Hodgkin-Katz equation extends this to several ions, and Hodgkin and Huxley (1952) modeled the action potential in the squid giant axon with voltage-dependent conductances.

Scale determines which physics dominates. Purcell (1977) showed that for micron-scale swimmers the Reynolds number is so low that inertia is negligible, so reciprocal motions cannot produce net travel; bacterial flagella and eukaryotic cilia circumvent this with non-reciprocal strokes. For larger organisms, insect flight involves unsteady aerodynamics such as leading-edge vortices. On the thermodynamic side, Schrodinger (1944) posed the question of how living matter avoids decay to equilibrium, and Prigogine's dissipative structures describe ordered states maintained by energy flux. Modern stochastic thermodynamics provides fluctuation theorems and bounds on the cost of precision, sensing, and copying in small systems.

## Role in Active Inference
This is the closest biology page to the physical origin of the free-energy principle. The principle is built on stochastic dynamics (Langevin and Fokker-Planck equations) and the notion of a non-equilibrium steady state, which are standard tools of biophysics and statistical physics. Neuronal biophysics, starting from Hodgkin-Huxley, underlies the neural mass and population models used in dynamic causal modeling. The derivations linking steady-state densities to Bayesian inference are, however, abstract and are not themselves empirical biophysical results, so it is better to treat biophysics as supplying the physical constraints and measurements to which the framework should be answerable.

## Related concepts
- [[knowledge_base/mathematics/non_equilibrium_thermodynamics]] — thermodynamics of driven, dissipative systems
- [[knowledge_base/mathematics/statistical_physics]] — the formal basis for fluctuations and ensembles
- [[knowledge_base/mathematics/langevin_dynamics]] — stochastic equations used for small-system dynamics
- [[knowledge_base/biology/insect_flight_dynamics]] — fluid mechanics of insect wings
- [[knowledge_base/biology/neuroscience]] — electrophysiology of excitable cells
- [[knowledge_base/biology/autopoiesis]] — organizational persistence of systems that are physically far from equilibrium

## References
- Hodgkin, A. L., and Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. Journal of Physiology.
- Purcell, E. M. (1977). Life at low Reynolds number. American Journal of Physics.
- Schrodinger, E. (1944). What is Life?
