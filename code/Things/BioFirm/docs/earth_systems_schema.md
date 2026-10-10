---
title: Earth Systems Active Inference Schema (Design Sketch)
type: design_notes
status: draft
created: 2025-01-01
updated: 2026-10-10
tags:
  - biofirm
  - earth_systems
  - multi_scale
  - schema
  - design
semantic_relations:
  - type: relates
    links:
      - "[[code/Things/BioFirm/BioFirm_README|BioFirm design notes]]"
      - "[[knowledge_base/BioFirm/biofirm_schema|BioFirm schema]]"
      - "[[knowledge_base/BioFirm/bioregional_state_space|Bioregional state space]]"
      - "[[knowledge_base/cognitive/hierarchical_inference|Hierarchical inference]]"
      - "[[knowledge_base/mathematics/active_inference_theory|Active inference theory]]"
---

# Earth Systems Active Inference Schema (Design Sketch)

> **Status: design sketch; no code in this repository.** This page proposes a vocabulary for modelling an earth system as nested Active Inference controllers. No classes (`NestedController`, `MultiScaleInference`, `ScaleCoupling`, `InterventionStrategy`, `EarthSystemViz`), data structures or configuration files described in earlier versions of this page exist in the repo. The YAML below is an illustrative schema, not a file that any tool reads. All numbers are example values, not calibrated to any site.

## Idea

Stewarding an ecosystem involves processes on very different timescales (hourly weather, yearly demography) and spatial extents (a patch, a landscape, a biome). The proposal is to give each scale its own generative model and homeostatic targets, and to let coarser scales supply contextual priors to finer ones while finer scales send prediction errors upward.

## Temporal scales (example)

```yaml
temporal_hierarchy:
  micro: {scale: hourly,  horizon: 24, update_frequency: hourly}
  meso:  {scale: daily,   horizon: 30, update_frequency: daily}
  macro: {scale: monthly, horizon: 12, update_frequency: weekly}
  meta:  {scale: yearly,  horizon: 10, update_frequency: monthly}
```

## Spatial scales (example)

```yaml
spatial_hierarchy:
  local:    {scale: patch,     dimensions_m: [100, 100],         resolution_m: 1}
  regional: {scale: landscape, dimensions_m: [10000, 10000],     resolution_m: 100}
  biome:    {scale: ecosystem, dimensions_m: [1000000, 1000000], resolution_m: 1000}
```

## State families

- **Ecological:** biodiversity, biomass, soil health, water cycles, energy flows, resilience.
- **Climate:** temperature, precipitation, wind, carbon cycling, radiative balance.
- **Human impact:** land use, resource extraction, pollution, restoration effort, social indicators.

## Homeostatic targets and interventions

Each indicator carries a target range (the preferred observations, C) rather than a single setpoint; for example a Shannon diversity index in a band, or natural land cover as a fraction. Interventions are grouped by domain: ecological restoration and management, climate mitigation and adaptation, and human policy and education.

## Scale coupling

Four couplings are worth parameterizing: upward influence (bottom-up evidence), downward control (top-down priors), lateral coupling within a scale, and temporal memory.

## Analysis ideas

Intervention efficiency, system resilience, and multi-scale trajectories before and after an intervention. Existing plotting utilities for the executable agents live under `code/tools/src/visualization/`.

## Related

- [[code/Things/BioFirm/BioFirm_README|BioFirm design notes]]
- [[knowledge_base/BioFirm/biofirm_framework|BioFirm framework]]
- [[knowledge_base/cognitive/homeostatic_control_theory|Homeostatic control theory]]
