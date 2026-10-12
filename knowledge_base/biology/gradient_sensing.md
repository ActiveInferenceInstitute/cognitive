---
title: Gradient Sensing
description: How cells and organisms detect spatial or temporal differences in chemical concentration to guide movement, covering bacterial chemotaxis, eukaryotic spatial sensing, ant trail following and physical limits.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - chemotaxis
  - sensory_biology
  - cell_signalling
semantic_relations:
  - type: relates
    links:
      - "[[trail_pheromones]]"
      - "[[trail_following]]"
      - "[[antennation]]"
      - "[[cell_migration]]"
      - "[[quorum_sensing]]"
      - "[[binding_sites]]"
---

# Gradient Sensing

## Overview
Gradient sensing is the detection of the direction and steepness of a concentration change, usually of a chemical, so that an organism can move toward food or a signal source or away from harm. Two strategies are used. In temporal sensing, a moving cell compares concentrations at successive times. In spatial sensing, a cell compares concentrations at different positions on its own surface or at two sensors, such as the two antennae of an insect.

## Mechanisms
- **Bacterial chemotaxis.** *Escherichia coli* is too small for a reliable difference across its body, so it measures change over time. Receptors bind attractant, a kinase cascade controls the direction of flagellar rotation, and the cell alternates smooth runs with random tumbles. When concentrations rise, runs are extended, which biases the walk up the gradient (Macnab and Koshland, 1972). Adaptation through methylation of the receptors resets sensitivity so that the cell responds to change in concentration over a wide background range.
- **Eukaryotic cells.** Amoebae such as *Dictyostelium* and white blood cells compare receptor occupancy across the cell, and amplify the small difference with internal signalling so that a front and back are established and the cell crawls up the gradient (Parent and Devreotes, 1999).
- **Ants.** Foragers follow trails of pheromone by sampling it with the antennae, and a difference between left and right helps steer along the trail (see [[trail_pheromones]], [[trail_following]] and [[antennation]]).

## Physical limits
Detection is limited by the finite number of molecules a sensor can count. Berg and Purcell (1977) showed that the precision with which a cell can estimate a concentration $c$ by averaging for time $\tau$ is limited by diffusive arrivals of molecules, with relative uncertainty scaling approximately as $(Dac\tau)^{-1/2}$ for a sensor of size $a$ and diffusion constant $D$. Larger sensors and longer averaging improve precision, at the cost of waiting time.

## Relation to Active Inference
Chemotaxis is a common illustration of active sensing, because the organism's own movement determines what it senses and a run-and-tumble policy samples the environment so as to reduce uncertainty about direction. Modelling the cell as minimising variational free energy, with attractant concentration as the hidden cause of receptor signals, is a modelling choice that reproduces such behaviour in simulation. It is a framework applied to the behaviour, not a finding about the cascade, which biochemical models already describe.

## Related concepts
- [[trail_pheromones]] — chemical signals that ants follow
- [[trail_following]] — the behaviour that uses gradient detection
- [[antennation]] — antennal sampling in insects
- [[cell_migration]] — directed movement of cells guided by gradients
- [[binding_sites]] — receptors where the molecular detection begins

## References
- Macnab, R. M., & Koshland, D. E. (1972). The gradient-sensing mechanism in bacterial chemotaxis. Proceedings of the National Academy of Sciences, 69, 2509-2512.
- Berg, H. C., & Purcell, E. M. (1977). Physics of chemoreception. Biophysical Journal, 20, 193-219.
- Parent, C. A., & Devreotes, P. N. (1999). A cell's sense of direction. Science, 284, 765-770.
- Berg, H. C. (2004). E. coli in Motion. Springer.
