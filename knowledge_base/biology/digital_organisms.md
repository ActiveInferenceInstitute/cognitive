---
title: Digital Organisms
description: Self-replicating computer programs that mutate and compete for computational resources, used as experimental systems for evolution, with Tierra and Avida as the main platforms.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - artificial_life
  - experimental_evolution
  - computational_biology
semantic_relations:
  - type: relates
    links:
      - "[[artificial_life]]"
      - "[[evolutionary_algorithms]]"
      - "[[natural_selection]]"
      - "[[cellular_automata]]"
      - "[[autopoiesis]]"
---

# Digital Organisms

## Overview
Digital organisms are computer programs that copy themselves, are subject to random mutation, and compete for limited resources such as processor time and memory. Because they satisfy the conditions for evolution by natural selection (heritable variation and differential reproduction), proponents argue that populations of them evolve in the full sense and are not merely a simulation of a separate biological system. This makes them a form of experimental evolution that runs in silico, with perfect records of ancestry and genotype and generation times of fractions of a second.

## Platforms
Tierra, created by Thomas Ray (1991), seeded a virtual machine with a self-copying ancestor; the population evolved shorter replicators, parasites that exploited the copy instructions of others, and hosts that resisted them. Avida, developed by Adami, Ofria, Wilke and colleagues, places programs on a spatial grid, gives each its own virtual processor, and rewards the execution of defined logic functions with extra processor time (Ofria and Wilke, 2004). Rewards are specified by the experimenter, but the means of computing the functions are not, so solutions must evolve.

## Findings
Digital evolution has been used to study the origin of complex features, mutational robustness, the evolution of evolvability, genome size, cooperation and the dynamics of adaptive radiations. In a widely cited Avida experiment, a complex function that required many instructions evolved through intermediate steps, some of which were neutral or deleterious when they first arose (Lenski et al., 2003). Such results show that complexity can arise by gradual selection without foresight, in a system that the researchers can fully inspect. A limit is that the instruction set and environment are artificial, so conclusions about biological details need independent support.

## Relation to Active Inference
The link is conceptual and weak. The free energy principle is a claim about systems that persist by maintaining their organisation, and digital organisms give a platform in which persistence, replication and the emergence of simple behaviours can be examined directly. Most digital-organism research makes no reference to Active Inference, and agents in these systems do not generally carry explicit generative models. Their value for the vault's themes is in showing how adaptation can arise from variation and selection alone.

## Related concepts
- [[artificial_life]] — the field of which digital organisms are a part
- [[evolutionary_algorithms]] — optimisation methods that borrow evolutionary operators
- [[natural_selection]] — the process the systems instantiate
- [[cellular_automata]] — another computational substrate for self-organisation
- [[autopoiesis]] — a theory of self-maintaining organisation

## References
- Ray, T. S. (1991). An approach to the synthesis of life. In C. G. Langton, C. Taylor, J. D. Farmer, & S. Rasmussen (Eds.), Artificial Life II (pp. 371-408). Addison-Wesley.
- Adami, C. (1998). Introduction to Artificial Life. Springer.
- Lenski, R. E., Ofria, C., Pennock, R. T., & Adami, C. (2003). The evolutionary origin of complex features. Nature, 423, 139-144.
- Ofria, C., & Wilke, C. O. (2004). Avida: A software platform for research in computational evolutionary biology. Artificial Life, 10, 191-229.
