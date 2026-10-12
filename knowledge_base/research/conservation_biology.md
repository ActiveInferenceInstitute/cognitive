---
title: Conservation Biology
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
      - "[[knowledge_base/research/environmental_science]]"
      - "[[knowledge_base/biology/conservation_biology_social_insects]]"
      - "[[knowledge_base/systems/ecosystem_management]]"
      - "[[knowledge_base/systems/adaptive_comanagement]]"
      - "[[knowledge_base/systems/traditional_ecological_knowledge]]"
      - "[[knowledge_base/biology/ecological_dynamics]]"
      - "[[knowledge_base/biology/population_genetics]]"
wikidata: Q641498
---

# Conservation Biology

## Overview

Conservation biology is the applied science of protecting biodiversity. Michael Soulé's 1985 article "What is conservation biology?" framed it as a crisis discipline: decisions must be made with incomplete information, on a deadline, with real costs of delay. It draws on population genetics, ecology, and increasingly on social science, because the causes of loss are human land use, harvesting, invasive species and climate change.

## Main questions

- **Population viability.** How large and connected a population must be to persist, and the effect of small size on genetic diversity and demographic chance.
- **Habitat and landscape.** Fragmentation, corridors, and where to place protected areas given limited budgets.
- **Monitoring.** Detecting declines in cryptic or rare species when surveys are imperfect, so that absence of detection is not absence of the species.
- **Management decisions.** Whether to act now, survey further, or stop managing, which is a sequential decision problem under uncertainty. Chadès and colleagues (2008) analysed exactly this problem for cryptic threatened species with a partially observable Markov decision process.
- **People and institutions.** Community-based conservation, and the role of traditional ecological knowledge.

## Connection to Active Inference

The connection is a modelling analogy at an early stage, not a program with field results. Conservation decisions are naturally formulated as partially observable decision problems, which is the class of problems Active Inference agents are built for. *Adaptive management*, the practice of treating interventions as experiments whose results update the model, matches the idea that the value of an action includes information gained as well as outcome achieved, which is what the epistemic and pragmatic terms of expected free energy represent. Imperfect detection fits an observation model in which the likelihood of detection given presence is below one.

Two cautions apply. The decision-theoretic literature in conservation already contains established tools, such as POMDP solvers and value-of-information analysis, and the free energy version should be compared with them and not assumed superior. Second, the preferences in such models, what should be conserved and at what cost, are value choices made by people, and are not given by the formalism.

The repository's social-insect pages tie these ideas to a specific taxonomic group, where colony-level organisation affects the way threats propagate through a population.

## In this repository

See [[knowledge_base/biology/conservation_biology_social_insects]], [[knowledge_base/research/environmental_science]], [[knowledge_base/systems/ecosystem_management]], [[knowledge_base/systems/adaptive_comanagement]], [[knowledge_base/systems/traditional_ecological_knowledge]], [[knowledge_base/biology/ecological_dynamics]] and [[knowledge_base/biology/population_genetics]]. A study path is [[active_inference_ecological_learning_path]].

## References

- Soulé, M. E. (1985). What is conservation biology? *BioScience*, 35, 727-734.
- Chadès, I., McDonald-Madden, E., McCarthy, M. A., Wintle, B., Linkie, M., & Possingham, H. P. (2008). When to stop managing or surveying cryptic threatened species. *Proceedings of the National Academy of Sciences*, 105, 13936-13940.

## Related Resources

- [[knowledge_base/research/environmental_science]]
- [[knowledge_base/biology/conservation_biology_social_insects]]
