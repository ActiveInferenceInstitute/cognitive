---
title: Evolutionary Systems
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - research
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/biology/evolutionary_biology]]"
      - "[[knowledge_base/free_energy_principle/README]]"
      - "[[knowledge_base/biology/evolutionary_dynamics]]"
      - "[[knowledge_base/biology/evolutionary_game_theory]]"
      - "[[knowledge_base/biology/niche_construction]]"
      - "[[knowledge_base/biology/population_genetics]]"
      - "[[knowledge_base/mathematics/evolutionary_algorithms]]"
      - "[[knowledge_base/free_energy_principle/biology/evolution]]"
      - "[[knowledge_base/research/cultural_evolution]]"
---

# Evolutionary Systems

## Overview

Evolutionary systems are populations of entities that vary, reproduce differentially and pass on their traits, so that composition changes over time. Evolutionary biology provides the standard account, and the mathematics of population genetics, evolutionary game theory and adaptive dynamics describe it. This page is concerned with how those dynamics relate to the free energy principle, which was formulated for the persistence of individual systems.

## Main concepts

- **Fitness landscapes.** Sewall Wright's 1932 metaphor of a surface of fitness over genotype space, on which populations climb peaks and can be trapped at local optima. In high dimensions the picture is subtler, and neutral networks and epistasis matter.
- **Adaptive radiation.** Diversification of a lineage into many ecological roles following ecological opportunity, as with Darwin's finches or cichlid fishes.
- **Niche construction.** Organisms modify their environments, altering the selection pressures on themselves and others (Odling-Smee, Laland and Feldman, 2003).
- **Evolutionary game theory.** Frequency-dependent selection and stable strategies, in which fitness depends on what others do.

## Connection to the free energy principle

The connection is an active theoretical programme, with several proposals of differing strength.

The first is that evolution can be described as inference. Campbell (2016) argued that natural selection can be formalised as Bayesian inference over a population's models of its environment, and Frank (2009) showed that selection can be written as a process that increases Fisher information. These re-descriptions are mathematically correct under their assumptions and do not by themselves supply new empirical predictions.

The second is Friston's 2013 account in "Life as we know it", in which systems that maintain a boundary and persist can be described as minimising free energy. Over evolutionary time, phenotypes that fail to keep their characteristic states do not persist, so the principle can be seen as a statement of what survival entails. Critics point out that this is close to a tautology, since any system that exists has by definition remained in its characteristic states, and that adaptive explanations need more than persistence.

The third is niche construction, which the free energy literature recasts as organisms shaping their environment so that it matches their generative model (Constant and colleagues, 2018; see [[knowledge_base/research/cultural_evolution]]). Established models of niche construction in population biology already capture the feedback, and the relationship between the two descriptions is still being worked out.

## In this repository

See [[knowledge_base/biology/evolutionary_biology]], [[knowledge_base/biology/evolutionary_dynamics]], [[knowledge_base/biology/evolutionary_game_theory]], [[knowledge_base/biology/niche_construction]], [[knowledge_base/biology/population_genetics]], [[knowledge_base/mathematics/evolutionary_algorithms]] and [[knowledge_base/free_energy_principle/biology/evolution]].

## References

- Wright, S. (1932). The roles of mutation, inbreeding, crossbreeding and selection in evolution. *Proceedings of the Sixth International Congress of Genetics*, 1, 356-366.
- Odling-Smee, F. J., Laland, K. N., & Feldman, M. W. (2003). *Niche Construction: The Neglected Process in Evolution*. Princeton University Press.
- Frank, S. A. (2009). Natural selection maximizes Fisher information. *Journal of Evolutionary Biology*, 22, 231-244.
- Campbell, J. O. (2016). Universal Darwinism as a process of Bayesian inference. *Frontiers in Systems Neuroscience*, 10, 49.
- Friston, K. (2013). Life as we know it. *Journal of the Royal Society Interface*, 10, 20130475.

## Related Resources

- [[knowledge_base/biology/evolutionary_biology]]
- [[knowledge_base/free_energy_principle/README]]
