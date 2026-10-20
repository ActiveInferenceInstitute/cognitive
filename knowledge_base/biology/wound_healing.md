---
external_mappings:
  - target: http://purl.obolibrary.org/obo/GO_0042060
    relation: exactMatch
    evidence: https://www.ebi.ac.uk/ols4/api/ontologies/go/terms?obo_id=GO:0042060
    rationale: >-
      The source defines repair after injury restoring tissue barrier function and
      structure, with haemostasis, inflammation, proliferation and remodelling. GO
      defines the series of events restoring integrity to damaged tissue following
      injury. Both denote wound healing as a process, not just cell migration or
      epithelial closure.
    reviewed_on: "2026-10-20"
title: Wound Healing
description: The coordinated sequence of haemostasis, inflammation, proliferation and remodelling by which tissue repairs an injury, with attention to collective cell migration in wound closure.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - tissue_repair
  - collective_migration
  - cell_biology
semantic_relations:
  - type: relates
    links:
      - "[[cell_migration]]"
      - "[[collective_migration]]"
      - "[[swarm_intelligence]]"
      - "[[morphogenesis]]"
      - "[[cell_proliferation]]"
wikidata: Q1509074
---

# Wound Healing

## Overview
Wound healing is the process by which injured tissue restores barrier function and structure. In mammalian skin it is commonly described in overlapping phases: haemostasis, in which clotting seals the injury; inflammation, in which immune cells clear debris and microbes; proliferation, which includes re-epithelialisation, formation of granulation tissue and growth of new vessels; and remodelling, in which the matrix is reorganised, often leaving a scar (Martin, 1997; Gurtner et al., 2008).

## Collective cell movement
Closure of an epithelial wound involves cells moving as connected sheets rather than as isolated individuals. Cells at the leading edge extend protrusions and exert traction on the substrate, while follower cells are pulled along and maintain cell-cell contacts; in some wounds a contractile actomyosin cable forms around the margin and helps pull the edge together. Friedl and Gilmour (2009) reviewed such collective migration in morphogenesis, regeneration and cancer, and highlighted features common to these settings: coordination through mechanical forces and chemical signals, a division between leader and follower cells, and cell-cell adhesion. Cells in the sheet respond to signals including growth factors, electrical fields and mechanical tension, and also to the free space created by the wound (see [[cell_migration]]).

## Variation and failure
Healing differs across tissues, ages and species. Embryonic wounds and those in some regenerating animals heal with little scarring, whereas adult skin tends to scar. Chronic wounds, as in diabetes or vascular disease, stall in the inflammatory phase. These observations are descriptive. The molecular determinants of scarring versus regeneration remain under investigation.

## Relation to Active Inference
The vault groups wound healing with swarm intelligence because both show group-level order from local interactions (see [[swarm_intelligence]], [[morphogenesis]]). Within Active Inference this connects to proposals that morphogenesis and repair can be modelled as cells inferring a target anatomy, but such proposals are still theoretical, and standard accounts of wound healing rest on cell mechanics and signalling.

## Related concepts
- [[cell_migration]] — movement of individual cells that sheets build on
- [[collective_migration]] — coordinated multicellular movement
- [[cell_proliferation]] — cell division supplying new tissue
- [[morphogenesis]] — pattern formation that repair partly recapitulates
- [[swarm_intelligence]] — coordination from local rules

## References
- Martin, P. (1997). Wound healing: aiming for perfect skin regeneration. Science, 276, 75-81.
- Gurtner, G. C., Werner, S., Barrandon, Y., & Longaker, M. T. (2008). Wound repair and regeneration. Nature, 453, 314-321.
- Friedl, P., & Gilmour, D. (2009). Collective cell migration in morphogenesis, regeneration and cancer. Nature Reviews Molecular Cell Biology, 10, 445-457.
