---
title: Information Thermodynamics
description: The study of how information acquired, stored and erased enters the laws of thermodynamics, including Landauer's principle, feedback control, generalised second laws, and connections to the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - thermodynamics
  - information_theory
  - stochastic_thermodynamics
semantic_relations:
  - type: relates
    links:
      - "[[landauer_principle]]"
      - "[[maxwell_demon]]"
      - "[[fluctuation_theorems]]"
      - "[[non_equilibrium_thermodynamics]]"
      - "[[mutual_information]]"
      - "[[work_extraction]]"
---

# Information Thermodynamics

## Overview
Information thermodynamics asks how information about a system, and the operations of measuring, storing and erasing it, can be incorporated into the second law. The central lesson is that information is a physical resource: knowledge of a system's state can be converted into extracted work, but acquiring, storing and resetting that knowledge has thermodynamic costs that restore the consistency of the second law.

## Key results
- **Landauer's principle**: erasing a bit in contact with a bath at temperature $T$ dissipates at least $k_BT\ln 2$ of heat (see [[landauer_principle]]).
- **Feedback and work extraction**: for a system controlled using measurement outcomes, the average extractable work obeys
$$\langle W_{\mathrm{ext}}\rangle\le -\Delta F + k_B T\, I,$$
where $\Delta F$ is the free energy change and $I$ is the mutual information between the measurement record and the system. Without information, $I=0$ and the usual bound is recovered. Sagawa and Ueda extended fluctuation relations such as the Jarzynski equality to this feedback setting.
- **Generalised second law**: the entropy production of a system together with its measuring device and memory is non-negative once the information flow between them is accounted for.
- **Information engines**: devices that rectify thermal fluctuations using measurement and feedback have been built experimentally with colloidal particles and single-electron circuits, converting information into work.

## Framework
Stochastic thermodynamics supplies the setting: small systems driven out of equilibrium are described by trajectories with fluctuating heat, work and entropy production, and fluctuation theorems constrain their statistics. Information enters through the mutual information between system and controller, which modifies these relations.

## Relation to Active Inference
The free energy principle partitions a system into internal and external states separated by a Markov blanket, and sensory and active states mediate information flow across it, which invites comparison with measurement and feedback in a Maxwell demon. The relation is interpretive. Variational free energy is an information-theoretic bound and not itself a measured thermodynamic free energy, and claims that Active Inference agents operate at thermodynamic limits are hypotheses. The connection is an active theoretical research area, and the exact correspondence depends on the physical assumptions made.

## Related concepts
- [[landauer_principle]] — the cost of erasure
- [[maxwell_demon]] — the paradigm of information-driven work extraction
- [[fluctuation_theorems]] — constraints on fluctuating work and entropy
- [[jarzynski_equality]] — a relation generalised to feedback control
- [[thermodynamic_computing]] — physical limits on computation
- [[markov_blanket]] — the boundary across which information flows in the free energy principle

## References
- Parrondo, J. M. R., Horowitz, J. M., & Sagawa, T. (2015). Thermodynamics of information. Nature Physics, 11, 131-139.
- Sagawa, T., & Ueda, M. (2010). Generalized Jarzynski equality under nonequilibrium feedback control. Physical Review Letters, 104, 090602.
- Landauer, R. (1961). Irreversibility and heat generation in the computing process. IBM Journal of Research and Development, 5, 183-191.
