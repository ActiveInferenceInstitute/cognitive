---
title: Cultural Dynamics
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
      - "[[knowledge_base/research/social_cognition]]"
      - "[[knowledge_base/research/cultural_evolution]]"
      - "[[knowledge_base/cognitive/cultural_cognition]]"
      - "[[knowledge_base/cognitive/social_learning]]"
      - "[[knowledge_base/cognitive/collective_behavior]]"
      - "[[knowledge_base/biology/evolutionary_game_theory]]"
      - "[[knowledge_base/systems/network_theory]]"
      - "[[knowledge_base/systems/social_systems]]"
      - "[[knowledge_base/research/cognitive_security]]"
---

# Cultural Dynamics

## Overview

Cultural dynamics concerns how the distribution of beliefs, norms and practices in a population changes over time and what drives the change. It is closely related to [[knowledge_base/research/cultural_evolution]], and the separation here is one of emphasis. Cultural evolution asks how culture is inherited and selected across generations. Cultural dynamics asks about the within-generation processes: how opinions spread through networks, how conventions settle, and how institutions form, persist and change.

## Main models

- **Opinion dynamics.** DeGroot's 1974 model of repeated averaging among neighbours gives conditions under which a network converges to consensus. Bounded-confidence and Axelrod-type models (1997) ask when diversity survives, and show polarisation or fragmentation depending on interaction rules.
- **Social contagion.** Diffusion of innovations and complex contagion, where adoption requires reinforcement from several contacts, which can differ strongly from simple epidemic spread.
- **Norms and conventions.** Game-theoretic accounts, following Lewis's analysis of convention, treat norms as equilibria of coordination games that are self-enforcing once established. Evolutionary game theory studies which equilibria are selected.
- **Institutions.** Rules and organisations that stabilise expectations, and the processes by which they are created, contested and revised.

These models are well developed and empirically uneven. Many are stylised and few have been validated against detailed longitudinal data.

## Connection to Active Inference

The connection is an active theoretical programme. The shared structure is that agents hold beliefs, observe others, and update. In Active Inference, a norm is a shared prior over joint behaviour, and a belief spreads when agents treat the beliefs of others as evidence with some assigned reliability, so precision weighting plays the role of trust. Conformity arises because predicting others correctly reduces uncertainty. Veissière and colleagues (2020) developed this view for culture, and the group-level models of Kaufmann and colleagues (2021) and Heins and colleagues (2022) show belief alignment in simulated populations of agents.

The phrase "collective free energy minimisation" should be read carefully. The free energy in question is a property of each agent's model, and a population-level quantity exists only if a specific collective model is defined. The simulated results demonstrate that the machinery can reproduce consensus, polarisation and echo chambers; they do not establish that real societies are well described this way, and the established opinion-dynamics models remain the benchmark against which such claims should be compared.

## In this repository

See [[knowledge_base/research/social_cognition]] for the individual-level mechanisms, [[knowledge_base/cognitive/cultural_cognition]], [[knowledge_base/cognitive/social_learning]], [[knowledge_base/cognitive/collective_behavior]], [[knowledge_base/biology/evolutionary_game_theory]], [[knowledge_base/systems/network_theory]] and [[knowledge_base/systems/social_systems]]. Information-integrity applications are in [[knowledge_base/research/cognitive_security]].

## References

- DeGroot, M. H. (1974). Reaching a consensus. *Journal of the American Statistical Association*, 69, 118-121.
- Axelrod, R. (1997). The dissemination of culture: a model with local convergence and global polarization. *Journal of Conflict Resolution*, 41, 203-226.
- Veissière, S. P. L., et al. (2020). Thinking through other minds: a variational approach to cognition and culture. *Behavioral and Brain Sciences*, 43, e90.

## Related Resources

- [[knowledge_base/research/social_cognition]]
- [[knowledge_base/research/cultural_evolution]]
