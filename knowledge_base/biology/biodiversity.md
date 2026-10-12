---
title: Biodiversity
description: "Biological diversity at the levels of genes, species and ecosystems, how it is measured with richness and entropy-based indices, and why it matters for stability and conservation."
type: biology
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - ecology
  - conservation
  - diversity
semantic_relations:
  - type: relates
    links:
      - "[[conservation_biology]]"
      - "[[ecosystem_services]]"
      - "[[community_ecology]]"
aliases:
  - biodiversity_assessment
  - Species_Diversity
wikidata: Q47041
---

# Biodiversity

## Overview
Biodiversity is the variety of life, conventionally described at three levels: genetic diversity within populations, species diversity within communities, and ecosystem diversity across landscapes. In the social-insect context of this knowledge base, ant and bee assemblages are common study groups, and the term often means species diversity of a local community.

## Definition and measurement
The simplest measure is species richness, the count of species present. Richness ignores abundance, so indices that weight by relative abundance $p_i$ are common. Shannon entropy is $H' = -\sum_i p_i \ln p_i$; four equally abundant species give $H' = \ln 4 \approx 1.386$, whereas the same four species at proportions 0.7, 0.1, 0.1, 0.1 give $H' \approx 0.940$. The exponential $e^{H'}$ converts this into an effective number of species: 4.0 in the even case and about 2.56 in the uneven one. Simpson's index $\sum p_i^2$ is the probability that two randomly drawn individuals share a species, and its inverse gives another effective number (about 1.92 for the uneven case). Whittaker's partition separates alpha diversity (within a site), beta diversity (turnover between sites) and gamma diversity (regional total).

Three recurring empirical themes are the species-area relationship, the latitudinal gradient in richness, and the relation between diversity and ecosystem function. Experimental work in grasslands and elsewhere has found that more diverse assemblages tend to have higher and more stable productivity, though the strength of this relationship varies with system and scale and is still debated.

## Threats and context
Principal drivers of loss are habitat destruction and [[habitat_fragmentation]], overexploitation, [[invasive_species]], pollution and [[climate_change_ecology]]. Diversity underpins many [[ecosystem_services]], which is the usual argument for conservation, alongside non-instrumental ethical arguments.

## Role in Active Inference
The connection is indirect. Within [[biofirm_framework]] and [[bioregional_state_space]], biodiversity indices serve as observable variables or preference targets in a model of a bioregion's condition. The Active Inference literature does not derive biodiversity from the free-energy principle; the use here is as a monitored state variable in a stewardship model, as in [[ecological_active_inference]].

## Related concepts
- [[conservation_biology]] — the applied discipline concerned with preserving diversity
- [[community_ecology]] — explains how local diversity is assembled and maintained
- [[restoration_ecology]] — rebuilding diversity in degraded systems
- [[ecosystem_services]] — benefits that depend on diversity
- [[information_theory]] — source of the entropy indices used above
