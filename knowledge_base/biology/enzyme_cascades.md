---
title: Enzyme Cascades
description: Sequences of enzyme reactions in which each activated enzyme activates many copies of the next, amplifying signals in clotting, phototransduction and kinase signalling, and giving switch-like responses.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - biochemistry
  - signalling
  - systems_biology
semantic_relations:
  - type: relates
    links:
      - "[[mapk_pathway]]"
      - "[[camp_pathway]]"
      - "[[biochemistry]]"
      - "[[systems_biology]]"
      - "[[feedback_loops]]"
      - "[[complex_systems_biology]]"
---

# Enzyme Cascades

## Overview
An enzyme cascade is a series of reactions in which each enzyme, once activated, activates the next enzyme in the series. Because one active enzyme molecule can modify many substrate molecules, each step multiplies the signal, so a small input can produce a large output. Cascades also allow regulation at several points and the integration of several inputs.

## Examples
- **Blood coagulation**: the waterfall or cascade model of Davie and Ratnoff and of Macfarlane described clotting as a sequence of proteolytic activations of inactive precursors (zymogens), culminating in thrombin converting fibrinogen to fibrin. The complement system of innate immunity is similarly organised.
- **Visual phototransduction**: a single photon activates rhodopsin, which activates many molecules of the G protein transducin, which in turn activate phosphodiesterase, so that a small number of photons produces a measurable change in the cell's electrical response.
- **Kinase cascades**: in the MAP kinase pathway (see [[mapk_pathway]]), kinases phosphorylate and activate one another in sequence.
- **Second-messenger pathways**: receptor activation leads through G proteins and enzymes to a messenger such as cAMP (see [[camp_pathway]]).

## Dynamics
Amplification is only one function. Goldbeter and Koshland showed that cycles of covalent modification, in which one enzyme adds and another removes a group such as phosphate, can display ultrasensitivity, a switch-like response in which output changes sharply over a narrow range of input. Stacking several such cycles can sharpen the response further, as shown experimentally for the MAP kinase cascade. Cascades can also contain feedback, which produces bistability or oscillations, and scaffold proteins can organise the components.

## Relation to Active Inference
Cascades are signal-processing circuits of the cell. There is no formal derivation of cascade behaviour from the Free Energy Principle. The hierarchical arrangement, in which activity at one level drives the next, resembles the layered message passing of hierarchical generative models, but this is an analogy and the biochemistry is explained by reaction kinetics.

## Related concepts
- [[mapk_pathway]] — a kinase cascade central to cell signalling
- [[camp_pathway]] — a second-messenger cascade acting through PKA
- [[systems_biology]] — modelling of cascades as dynamical systems
- [[feedback_loops]] — regulation that shapes cascade dynamics
- [[gene_regulatory_networks]] — downstream targets of many signalling cascades

## References
- Davie, E. W., & Ratnoff, O. D. (1964). Waterfall sequence for intrinsic blood clotting. Science, 145, 1310-1312.
- Goldbeter, A., & Koshland, D. E. (1981). An amplified sensitivity arising from covalent modification in biological systems. Proceedings of the National Academy of Sciences, 78, 6840-6844.
- Berg, J. M., Tymoczko, J. L., Gatto, G. J., & Stryer, L. (2019). Biochemistry (9th ed.). W. H. Freeman.
