---
title: Evolution of Eusociality
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - eusociality
  - social_evolution
  - kin_selection
  - social_insects
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/biology/ant_colony_organization]]"
      - "[[knowledge_base/biology/evolutionary_biology]]"
      - "[[knowledge_base/biology/evolutionary_game_theory]]"
      - "[[knowledge_base/biology/myrmecology]]"
wikidata: Q5418647
---

# Evolution of Eusociality

## Overview
Eusociality is the most elaborate form of animal social organization, defined by three traits: overlapping generations, cooperative care of brood, and a reproductive division of labor in which some individuals (workers) forgo reproduction. It has evolved repeatedly in ants, several bee and wasp lineages (all Hymenoptera), termites, and a few other taxa such as snapping shrimp (Synalpheus) and the mole-rats Heterocephalus glaber and Fukomys damarensis. The central question is how sterile or non-reproducing castes can be favored by selection.

## Mechanism
Hamilton's rule (1964) states that a trait favoring helping is selected when rB > C, where r is genetic relatedness between actor and recipient, B the reproductive benefit to the recipient, and C the cost to the actor. Haplodiploidy initially looked like an explanation, because full sisters share on average 0.75 of their genes. Trivers and Hare (1976) showed that this advantage is offset when sex-ratio investment is equal, and termites, which are diploid, are fully eusocial, so haplodiploidy is neither necessary nor sufficient.

The monogamy hypothesis (Boomsma 2009) holds that lifelong monogamy, which makes r between siblings and offspring equal, was the ancestral state in the independent origins of obligate eusociality in insects; later polyandry or polygyny evolved after the worker caste was already irreversible. Ecological pre-adaptations also matter: defensible, food-rich nests (fortress defense), parental care, and a protected brood. Nowak, Tarnita and Wilson (2010) argued for a model based on group-level selection acting on these ecological factors rather than inclusive fitness; this was contested in detail by many inclusive-fitness researchers, and the dispute has not been fully resolved in the sense of reaching consensus. Once a colony has a sterile worker caste it is treated as a major transition: selection acts at the colony level, which is why colonies are called superorganisms.

## Role in Active Inference
The link is indirect. Eusociality is an evolutionary account of why colony-level units exist; Active Inference is a theory of how a unit with a boundary maintains itself. If a colony is treated as an agent with its own boundary, then a colony-level generative model (the information held in pheromone fields, nest structure, and worker behavior thresholds) becomes a meaningful object of study, which is the assumption behind the Ant_Colony Thing in this repository. Whether colony-level inference follows rigorously from individual-level free-energy minimization is an open modeling question, not a result.

## Related concepts
- [[knowledge_base/biology/ant_colony_organization]] — the caste and division-of-labor outcome of eusocial evolution in ants
- [[knowledge_base/biology/bee_colony_organization]] — eusociality in bees, including both primitively and advanced social species
- [[knowledge_base/biology/evolutionary_game_theory]] — formal tools for cooperation and conflict between relatives
- [[knowledge_base/biology/niche_construction]] — nest building as a shared environmental modification
- [[knowledge_base/systems/task_allocation]] — the operational division of labor within a eusocial colony

## References
- Hamilton, W. D. (1964). The genetical evolution of social behaviour I and II. Journal of Theoretical Biology.
- Trivers, R. L., and Hare, H. (1976). Haplodiploidy and the evolution of the social insects. Science.
- Wilson, E. O. (1971). The Insect Societies.
- Boomsma, J. J. (2009). Lifetime monogamy and the evolution of eusociality. Philosophical Transactions of the Royal Society B.
- Nowak, M. A., Tarnita, C. E., and Wilson, E. O. (2010). The evolution of eusociality. Nature.
