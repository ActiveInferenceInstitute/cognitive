---
title: Canonical Circuits
description: The idea that neocortex repeats a common microcircuit of layered excitatory and inhibitory cells, its main connections, the caveats, and its use as the anatomical basis for predictive coding.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - cortical_microcircuit
  - neuroanatomy
  - predictive_coding
semantic_relations:
  - type: relates
    links:
      - "[[cortical_columns]]"
      - "[[hierarchical_processing]]"
      - "[[deep_pyramidal_cells]]"
      - "[[superficial_pyramidal_cells]]"
      - "[[interneurons]]"
      - "[[predictive_coding]]"
---

# Canonical Circuits

## Overview
A canonical circuit, or canonical microcircuit, is a stereotyped pattern of connections among cortical cell types that is proposed to recur across areas and species of neocortex. The idea builds on the observation that cortex has a similar six-layer organisation almost everywhere, and it suggests that different areas perform variants of a common computation on different inputs. Douglas, Martin and Whitteridge (1989) offered an early formal version, and the concept is reviewed by Douglas and Martin (2004).

## Typical connectivity
- **Input.** Thalamic afferents terminate mainly in layer 4, which projects to layers 2/3.
- **Intracortical flow.** Layers 2/3 project to layer 5, and layer 5 and 6 cells feed back to the earlier layers, with strong recurrent excitation among pyramidal cells.
- **Output.** Layer 2/3 cells send feedforward projections to higher areas, layer 5 cells project to subcortical targets, and layer 6 cells project back to the thalamus.
- **Inhibition.** Several classes of interneuron, including fast-spiking basket cells, regulate gain and timing (see [[interneurons]]).
- **Between areas.** Forward connections from superficial layers target layer 4 of the next area, while backward connections avoid layer 4 and arise from both superficial and deep layers (Felleman & Van Essen, 1991).

## Caveats
Areas differ in layer thickness, cell density and connectivity. Motor cortex has little or no layer 4, and primary visual cortex subdivides it. "Canonical" is therefore an abstraction that captures common features, not a claim of uniformity. Lists of cell types and connection strengths have also been revised as methods improved.

## Role in Active Inference
Bastos and colleagues (2012) mapped the message passing of predictive coding onto the canonical microcircuit. In their proposal, superficial pyramidal cells encode prediction errors and send them forward, deep pyramidal cells encode expectations and send predictions backward, and inhibitory interneurons support the precision weighting of errors. This is an active research programme with some supporting physiological evidence, but the cell-type assignments are contested and other circuit-level accounts of prediction exist.

## Related concepts
- [[cortical_columns]] — the vertical organisation in which the circuit is embedded
- [[hierarchical_processing]] — how areas are ordered by forward and backward connections
- [[deep_pyramidal_cells]] — proposed sources of top-down predictions
- [[superficial_pyramidal_cells]] — proposed sources of bottom-up prediction errors
- [[predictive_coding]] — the computational scheme mapped onto the circuit

## References
- Douglas, R. J., Martin, K. A. C., & Whitteridge, D. (1989). A canonical microcircuit for neocortex. Neural Computation, 1, 480-488.
- Douglas, R. J., & Martin, K. A. C. (2004). Neuronal circuits of the neocortex. Annual Review of Neuroscience, 27, 419-451.
- Bastos, A. M., Usrey, W. M., Adams, R. A., Mangun, G. R., Fries, P., & Friston, K. J. (2012). Canonical microcircuits for predictive coding. Neuron, 76, 695-711.
- Felleman, D. J., & Van Essen, D. C. (1991). Distributed hierarchical processing in the primate cerebral cortex. Cerebral Cortex, 1, 1-47.
