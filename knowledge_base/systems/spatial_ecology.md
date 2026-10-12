---
title: Spatial Ecology
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - systems
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/systems/ecosystem_management]]"
      - "[[knowledge_base/systems/environmental_applications]]"
      - "[[knowledge_base/biology/ecological_networks]]"
      - "[[knowledge_base/BioFirm/bioregional_state_space]]"
wikidata: Q3578586
---

# Spatial Ecology

## Overview

Spatial ecology studies how the arrangement of habitat in space affects the distribution, movement and persistence of organisms and the flow of materials. Landscapes are described as patches of habitat embedded in a matrix of other cover, linked or separated by corridors and barriers. The established core is statistical and mechanistic ecology. Linking it to Active Inference is a proposal developed in this repository's BioFirm material, and the connection is mostly one of shared structure (nested boundaries, multi-scale state estimation), not an established method.

## Main Methods

**Metapopulation theory.** Levins's model describes a regional population as a set of local populations that go extinct and recolonise, so persistence depends on the balance of those two rates and on the patch network.

**Habitat connectivity.** Landscapes can be represented as graphs, with patches as nodes and dispersal links as weighted edges, and analysed for connectivity and critical nodes ([[knowledge_base/mathematics/graph_theory]], [[knowledge_base/biology/ecological_networks]]). Circuit theory (McRae and colleagues 2008) treats the landscape as a resistance surface and models movement as current flow, which captures multiple routes better than a single least-cost path.

**Species distribution models.** These relate observed occurrence to environmental layers to map suitability. Maximum-entropy methods (Phillips et al. 2006) find the least-constrained distribution consistent with the observed environmental feature averages. Results depend on sampling bias in presence records, and a suitability map is not a map of occupancy.

**Landscape dynamics.** Disturbance, succession and climate shift the habitat template over time, so spatial models often need a dynamic component ([[knowledge_base/biology/ecological_dynamics]]).

## Connection to Active Inference

Two links are concrete. First, an organism moving through a landscape is a foraging agent with a limited view, which fits epistemic foraging in a spatial setting ([[knowledge_base/cognitive/epistemic_foraging]], [[knowledge_base/cognitive/spatial_cognition]]). Second, multi-scale ecological systems can be described as nested boundaries, with local ecosystems inside landscapes inside regions, which BioFirm formalises as nested Markov blankets over a bioregional state space ([[knowledge_base/BioFirm/bioregional_state_space]], [[knowledge_base/mathematics/markov_blanket]]). Whether an ecosystem or landscape has a Markov blanket in the technical sense, rather than a convenient boundary, is a modelling assumption that needs justification case by case. The maximum-entropy relation is also loose: it shares information-theoretic foundations with free-energy methods but is a different procedure.

## In This Repository

Related pages are [[knowledge_base/systems/ecosystem_management]], [[knowledge_base/systems/environmental_applications]], [[knowledge_base/systems/environmental_monitoring]], [[knowledge_base/biology/niche_construction]] and [[knowledge_base/BioFirm/ecological_active_inference]]. The code tree contains no landscape, connectivity or species-distribution model; `code/Things/BioFirm` is documentation, including an earth-systems schema.

## References

- Phillips, S. J., Anderson, R. P., Schapire, R. E. (2006). Maximum entropy modeling of species geographic distributions. *Ecological Modelling*.
- McRae, B. H., Dickson, B. G., Keitt, T. H., Shah, V. B. (2008). Using circuit theory to model connectivity in ecology, evolution, and conservation. *Ecology*.
- Levins, R. (1969). Some demographic and genetic consequences of environmental heterogeneity for biological control. *Bulletin of the Entomological Society of America*.

## Related Resources

- [[knowledge_base/systems/ecosystem_management]]
- [[knowledge_base/systems/environmental_applications]]
