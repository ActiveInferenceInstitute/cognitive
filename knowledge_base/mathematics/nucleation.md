---
title: Nucleation
description: The onset of a new phase at localised seed regions in a metastable system, covered through classical nucleation theory, the critical radius, the free energy barrier and the distinction between homogeneous and heterogeneous nucleation.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - phase_transitions
  - statistical_physics
  - metastability
semantic_relations:
  - type: relates
    links:
      - "[[phase_transitions]]"
      - "[[thermodynamics]]"
      - "[[statistical_physics]]"
      - "[[pattern_formation]]"
      - "[[order_parameters]]"
wikidata: Q909022
---

# Nucleation

## Overview
Nucleation is the first step in the formation of a new thermodynamic phase from a metastable one, such as a droplet in supersaturated vapour or a crystal in a supercooled liquid. A metastable phase is stable against small fluctuations but not against a sufficiently large localised cluster of the new phase. Transformation proceeds through the stochastic appearance of such a cluster, which then grows.

## Classical nucleation theory
Consider a spherical cluster of radius $r$ of the stable phase inside the metastable phase. Its free energy change has a volume term, favourable when the new phase has lower free energy density by $\Delta g>0$, and a surface term from the interfacial tension $\sigma$:
$$\Delta G(r)=-\tfrac43\pi r^{3}\Delta g+4\pi r^{2}\sigma .$$
The two terms compete. Setting $d\Delta G/dr=0$ gives the critical radius and barrier
$$r^{*}=\frac{2\sigma}{\Delta g},\qquad \Delta G^{*}=\frac{16\pi\sigma^{3}}{3\,\Delta g^{2}} .$$
Clusters smaller than $r^{*}$ tend to dissolve and larger ones tend to grow. The nucleation rate per unit volume is approximately $J\approx J_0\exp(-\Delta G^{*}/k_BT)$, so it depends very steeply on the driving force, because $\Delta G^{*}$ falls as the degree of supersaturation or undercooling rises. The kinetic prefactor $J_0$ includes the rate at which molecules attach to a cluster, with the treatment going back to Becker and Döring (1935).

Nucleation is **homogeneous** when clusters form spontaneously in the bulk, and **heterogeneous** when they form on surfaces, impurities or container walls. A surface that the new phase wets lowers the barrier by a factor that depends on the contact angle, which is why most nucleation in practice is heterogeneous. Classical theory treats small clusters as having the bulk density and sharp interfaces, an approximation that can fail badly for the very small clusters involved, and its quantitative predictions often differ from experiment by orders of magnitude.

## Relation to Active Inference
Nucleation has no derivation within Active Inference. The connection is conceptual: models of self-organisation and emergence describe order arising from seeds that exceed a critical size, and symmetry-breaking transitions are used as an analogy for sudden changes in neural or behavioural dynamics. These uses are metaphors, not established results.

## Related concepts
- [[phase_transitions]] — the broader class of transformations
- [[thermodynamics]] — free energy comparison between phases
- [[statistical_physics]] — fluctuations that produce critical clusters
- [[pattern_formation]] — growth of ordered structure
- [[order_parameters]] — variables distinguishing the phases

## References
- Becker, R., & Döring, W. (1935). Kinetische Behandlung der Keimbildung in übersättigten Dämpfen. Annalen der Physik, 416, 719-752.
- Debenedetti, P. G. (1996). Metastable Liquids: Concepts and Principles. Princeton University Press.
- Kelton, K. F., & Greer, A. L. (2010). Nucleation in Condensed Matter: Applications in Materials and Biology. Pergamon.
