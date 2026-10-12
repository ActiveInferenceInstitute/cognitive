---
title: Thermodynamic Computing
description: Computing viewed through its physical energy cost and implemented with thermal or stochastic physical processes, including Landauer's bound, reversible computation, and relations to the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - thermodynamics
  - computation
  - information
semantic_relations:
  - type: relates
    links:
      - "[[landauer_principle]]"
      - "[[information_thermodynamics]]"
      - "[[thermodynamics]]"
      - "[[neuromorphic_computing]]"
      - "[[entropy_production]]"
wikidata: Q107065377
---

# Thermodynamic Computing

## Overview
Thermodynamic computing covers two related ideas. The first treats computation as a physical process that must obey thermodynamic limits, so that information processing has a minimum energy and entropy cost. The second builds computers that exploit physical dissipation and thermal noise, for example by sampling from probability distributions with stochastic hardware instead of fighting fluctuations.

## Physical limits
Landauer's principle states that erasing one bit of information in an environment at temperature $T$ dissipates at least
$$Q\ge k_B T\ln 2$$
of heat, where $k_B$ is Boltzmann's constant. The bound applies to logically irreversible operations, which map several inputs to one output and thereby reduce the entropy of the information-bearing degrees of freedom. Bennett showed that any computation can in principle be made logically reversible, by keeping a record of intermediate steps, so that no heat need be dissipated by the logic itself, and that the unavoidable cost arises when memory is erased. Practical processors operate many orders of magnitude above the bound, so it is a fundamental limit and not a current engineering constraint.

## Physical implementations
Proposals include probabilistic bits whose state fluctuates randomly under tunable bias, used for sampling and optimisation, and analogue devices whose relaxation to thermal equilibrium solves linear algebra or inference problems. Neuromorphic and biological systems are sometimes discussed as examples of efficient computation near the limits set by noise, since a brain operates on the order of tens of watts.

## Relation to Active Inference
The free energy principle uses the vocabulary of thermodynamics, because variational free energy has the same functional form as a thermodynamic free energy, and it is often discussed with Landauer's principle to argue that inference carries an energetic cost. The strength of the link varies. The formal correspondence between the two free energies is a mathematical identity of form, while claims that biological self-organisation minimises thermodynamic cost through inference are hypotheses of an active research programme, not established results. Information-theoretic efficiency and thermodynamic efficiency are related but are not the same quantity.

## Related concepts
- [[landauer_principle]] — the minimum cost of erasing a bit
- [[information_thermodynamics]] — the wider link between information and thermodynamics
- [[maxwell_demon]] — the thought experiment that motivated the analysis
- [[entropy_production]] — the dissipation that bounds are expressed in
- [[neuromorphic_computing]] — hardware modelled on neural efficiency
- [[work_extraction]] — extracting work using information

## References
- Landauer, R. (1961). Irreversibility and heat generation in the computing process. IBM Journal of Research and Development, 5, 183-191.
- Bennett, C. H. (1973). Logical reversibility of computation. IBM Journal of Research and Development, 17, 525-532.
- Bennett, C. H. (1982). The thermodynamics of computation: a review. International Journal of Theoretical Physics, 21, 905-940.
