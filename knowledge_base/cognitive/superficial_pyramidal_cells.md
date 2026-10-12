---
title: Superficial Pyramidal Cells
description: Pyramidal neurons in cortical layers II and III, their feedforward connections, and the predictive coding proposal that they encode prediction errors, with the status of the supporting evidence.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - cortical_microcircuit
  - predictive_coding
  - neuroanatomy
semantic_relations:
  - type: relates
    links:
      - "[[prediction_error]]"
      - "[[predictive_coding]]"
      - "[[deep_pyramidal_cells]]"
      - "[[cortical_columns]]"
      - "[[precision_weighting]]"
---

# Superficial Pyramidal Cells

## Overview
Superficial pyramidal cells are excitatory neurons whose cell bodies lie in the upper layers of the neocortex, mainly layers II and III. They are characterised by a triangular soma, a prominent apical dendrite extending toward the cortical surface, and basal dendrites. Anatomically, they are the main source of feedforward corticocortical projections to higher areas, together with some layer IV spiny neurons, whereas feedback projections originate mainly from deep layers V and VI. This asymmetry underlies the laminar rules of cortical hierarchies described by Felleman and Van Essen (1991).

## Physiology
Superficial layers show strong local recurrent connections and receive input from layer IV, the main thalamic recipient in sensory areas. In recordings from visual cortex, activity in superficial layers has been associated with higher-frequency gamma-band synchronisation, whereas deep layers show more power in alpha and beta bands. Bastos et al. (2015) reported that feedforward influences between visual areas were stronger in the gamma (and theta) range and feedback influences in the alpha-beta range, a pattern consistent with the laminar origin of these pathways.

## Role in predictive coding
Predictive coding schemes divide cortical populations into units that represent expectations and units that represent prediction errors. In the canonical microcircuit proposed by Bastos et al. (2012), superficial pyramidal cells encode prediction errors and send them forward to the next level of the hierarchy, while deep pyramidal cells encode conditional expectations and send predictions backward (see [[deep_pyramidal_cells]]). Gain on error units is proposed to be set by precision, possibly through neuromodulators and inhibitory interneurons (see [[precision_weighting]]).

The empirical status is mixed. The anatomy of forward and backward connections and the frequency asymmetry are well supported, whereas direct evidence that superficial cells encode prediction error in the strict sense is less secure. Some studies have found enhanced responses to unexpected stimuli in superficial layers, but interpretation varies and other schemes assign different roles to the same cells. The mapping should therefore be regarded as a hypothesis within the predictive coding programme, not an established fact about cortex.

## Related concepts
- [[prediction_error]] — the signal proposed to be carried by these cells
- [[predictive_coding]] — the scheme that assigns this role
- [[deep_pyramidal_cells]] — the proposed carriers of expectations
- [[cortical_columns]] — the laminar organisation in which both populations sit
- [[beta_oscillations]] — a rhythm associated with feedback in deep layers
- [[precision_weighting]] — gain control on error units

## References
- Felleman, D. J., & Van Essen, D. C. (1991). Distributed hierarchical processing in the primate cerebral cortex. Cerebral Cortex, 1, 1-47.
- Bastos, A. M., Usrey, W. M., Adams, R. A., Mangun, G. R., Fries, P., & Friston, K. J. (2012). Canonical microcircuits for predictive coding. Neuron, 76, 695-711.
- Bastos, A. M., Vezoli, J., Bosman, C. A., Schoffelen, J.-M., Oostenveld, R., Dowdall, J. R., De Weerd, P., Kennedy, H., & Fries, P. (2015). Visual areas exert feedforward and feedback influences through distinct frequency channels. Neuron, 85, 390-401.
