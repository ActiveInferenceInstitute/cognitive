---
title: Work Extraction
description: How much useful work can be drawn from a system in contact with a heat bath, the free energy and information bounds on it, and its parallels to free energy in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - thermodynamics
  - information_thermodynamics
  - free_energy
semantic_relations:
  - type: relates
    links:
      - "[[thermodynamics]]"
      - "[[information_thermodynamics]]"
      - "[[maxwell_demon]]"
      - "[[landauer_principle]]"
      - "[[jarzynski_equality]]"
---

# Work Extraction

## Overview
Work extraction asks how much useful, ordered energy can be obtained from a system as it relaxes toward equilibrium or is driven through a protocol while coupled to a thermal reservoir. The second law of thermodynamics limits the answer, and the limit is expressed through free energy rather than total energy. The topic also underlies the analysis of measurement and feedback, where information about a system can be converted into extra extractable work.

## Free energy bound
For a system at temperature $T$ that changes between two equilibrium states in contact with a bath, the average work extracted obeys
$$\langle W_{\mathrm{ext}}\rangle \le -\Delta F, \qquad F = U - TS,$$
where $F$ is the Helmholtz free energy. Equality holds only for a reversible process. For finite systems, the Jarzynski equality $\langle e^{-\beta W}\rangle = e^{-\beta \Delta F}$, with $W$ the work done on the system and $\beta = 1/k_B T$, implies $\langle W\rangle \ge \Delta F$ by Jensen's inequality, while individual trajectories can transiently violate the bound.

## Information as a resource
A measurement followed by feedback can extract more than $-\Delta F$. In the Szilard engine, one bit of information about which half of a box holds a single particle allows up to $k_B T \ln 2$ of work per cycle. Generalised second-law results bound the feedback gain by $k_B T$ times the mutual information $I$ between the measurement record and the system state, so that $\langle W_{\mathrm{ext}}\rangle \le -\Delta F + k_B T\, I$. The apparent paradox of Maxwell's demon is resolved by accounting for the memory: resetting it costs at least $k_B T \ln 2$ per erased bit (see [[landauer_principle]] and [[maxwell_demon]]).

## Relation to Active Inference
The link to Active Inference is formal and partly analogical. Variational free energy has the same structure as a thermodynamic free energy, an expected energy term minus an entropy term, and both serve as bounds that govern relaxation. Stochastic thermodynamics has been applied to systems with a Markov blanket to relate belief updating to dissipation, and this is an active research direction. The free energy minimised by an agent is nonetheless an information-theoretic quantity over beliefs, and nothing in the framework says organisms maximise extracted work.

## Related concepts
- [[thermodynamics]] — the second-law framework that sets the bounds
- [[information_thermodynamics]] — treats information as a thermodynamic resource
- [[maxwell_demon]] — the thought experiment that motivates feedback extraction
- [[landauer_principle]] — the erasure cost that closes the accounting
- [[jarzynski_equality]] — fluctuation relation linking work to free energy differences
- [[free_energy_principle]] — uses variational rather than thermodynamic free energy

## References
- Landauer, R. (1961). Irreversibility and heat generation in the computing process. IBM Journal of Research and Development, 5, 183-191.
- Jarzynski, C. (1997). Nonequilibrium equality for free energy differences. Physical Review Letters, 78, 2690-2693.
- Parrondo, J. M. R., Horowitz, J. M., & Sagawa, T. (2015). Thermodynamics of information. Nature Physics, 11, 131-139.
