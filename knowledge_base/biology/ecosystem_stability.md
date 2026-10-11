---
title: Ecosystem Stability
description: The ability of ecosystems and communities to persist and recover under perturbation, its components such as resistance and resilience, and the long-running debate on how diversity and complexity relate to it.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - ecology
  - stability
  - resilience
semantic_relations:
  - type: relates
    links:
      - "[[ecological_networks]]"
      - "[[resilience]]"
      - "[[biodiversity]]"
      - "[[stability_analysis]]"
      - "[[ecosystem_processes]]"
      - "[[ecological_dynamics]]"
---

# Ecosystem Stability

## Overview
Ecosystem stability is a collective name for several distinct properties describing how populations, communities or ecosystem functions respond to disturbance. Using the word without specifying which property is meant is a common source of confusion. Holling (1973) separated stability, the tendency to return to equilibrium after a disturbance, from resilience, the amount of disturbance a system can absorb before shifting to a different regime.

## Components
- **Resistance.** How little a system changes when disturbed.
- **Resilience.** How quickly it returns to its reference state (engineering resilience) or how large a disturbance it can tolerate before changing regime (Holling's ecological resilience).
- **Constancy or variability.** The temporal variability of a quantity such as biomass.
- **Persistence.** Whether species or properties survive over time.

Different measures can disagree, and the answer depends on the level considered, from a single population to total community productivity.

## Complexity and diversity
Intuition once held that richer and more connected communities are more stable. May (1972) challenged this with random community matrices. For $S$ species interacting with connectance $C$ and interaction strengths of typical size $\sigma$, relative to self-regulation of unit strength, a random system is likely to be stable only if $\sigma\sqrt{SC}<1$, so stability declines as size or connectance rises. Real food webs are not random, and non-random structure, such as weak interactions and compartments, can restore stability. Empirical work finds that diversity tends to stabilise aggregate community properties, such as total biomass, while the stability of individual populations may fall, and so the answer depends on the definition (Tilman, 1996; McCann, 2000).

## Relation to Active Inference
The relationship is partly formal and partly metaphorical. Stability analysis based on linearising dynamics around a fixed point is the same mathematics used for steady states in the Free Energy Principle (see [[stability_analysis]]). Whether an ecosystem should be treated as an agent that infers is contested; the programme of ecological Active Inference mostly models individual organisms interacting with their niche (see [[ecological_active_inference]]).

## Related concepts
- [[ecological_networks]] — the interaction structure that affects stability
- [[resilience]] — capacity to absorb disturbance and reorganise
- [[biodiversity]] — variety of life, often linked to stability
- [[stability_analysis]] — mathematical treatment of equilibria
- [[species_richness]] — a basic component of diversity

## References
- Holling, C. S. (1973). Resilience and stability of ecological systems. Annual Review of Ecology and Systematics, 4, 1-23.
- May, R. M. (1972). Will a large complex system be stable? Nature, 238, 413-414.
- McCann, K. S. (2000). The diversity-stability debate. Nature, 405, 228-233.
- Tilman, D. (1996). Biodiversity: population versus ecosystem stability. Ecology, 77, 350-363.
