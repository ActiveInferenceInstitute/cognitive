---
title: Early LTP
description: The initial phase of long-term potentiation, lasting roughly one to three hours, which depends on NMDA receptor activation and existing proteins and is independent of new protein synthesis.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - synaptic_plasticity
  - ltp
  - memory
semantic_relations:
  - type: relates
    links:
      - "[[ltp]]"
      - "[[late_ltp]]"
      - "[[nmda_receptors]]"
      - "[[synaptic_plasticity]]"
      - "[[synaptic_tagging]]"
---

# Early LTP

## Overview
Long-term potentiation (LTP) is a persistent strengthening of synaptic transmission after patterned activity, first described by Bliss and Lømo in 1973 in the rabbit hippocampus. It is often divided into phases. Early LTP, or E-LTP, is the first phase, beginning within seconds of induction and decaying over about one to three hours unless it is converted into a lasting form. It does not require new gene transcription or protein synthesis, which distinguishes it from late LTP.

## Induction
At many excitatory synapses, notably the CA1 region of the hippocampus, induction depends on coincident presynaptic glutamate release and postsynaptic depolarisation. This relieves the magnesium block of NMDA receptors and allows calcium to enter the dendritic spine. The size and timing of the calcium rise determine whether potentiation (LTP) or depression results.

## Expression
- **Kinase activation.** Calcium activates calcium/calmodulin-dependent protein kinase II (CaMKII), which becomes persistently active through autophosphorylation, and other kinases such as protein kinase C.
- **AMPA receptor changes.** Phosphorylation increases the conductance of existing AMPA receptors, and receptors from intracellular pools are inserted into the postsynaptic membrane, raising the response to released glutamate (Malenka and Bear, 2004; Huganir and Nicoll, 2013).
- **Presynaptic contributions.** At some synapses, changes in release probability may also contribute, a point still debated.

Because these changes use proteins already present, they can occur quickly and be induced locally, but they fade unless stabilised. Strong or repeated stimulation triggers signalling to the nucleus and local protein synthesis that maintain late LTP, and the synaptic tagging and capture hypothesis proposes that a tagged synapse can capture newly made plasticity-related proteins.

## Relation to Active Inference
In Active Inference and related Bayesian accounts, synaptic efficacy encodes the parameters of the generative model, and Hebbian-like changes can emerge from gradient descent on free energy in neural-network implementations. Early LTP provides a candidate biological process for fast changes in these parameters. The mapping is a theoretical interpretation, and no experiment has tied a specific term of the free energy to a particular LTP phase.

## Related concepts
- [[ltp]] — the general phenomenon
- [[late_ltp]] — the protein-synthesis-dependent phase
- [[nmda_receptors]] — the coincidence detectors for induction
- [[synaptic_plasticity]] — the wider class of activity-dependent changes
- [[synaptic_tagging]] — the mechanism proposed for converting early to late LTP
- [[hebbian_learning]] — the learning principle LTP illustrates

## References
- Bliss, T. V. P., & Lømo, T. (1973). Long-lasting potentiation of synaptic transmission in the dentate area of the anaesthetized rabbit following stimulation of the perforant path. Journal of Physiology, 232, 331-356.
- Malenka, R. C., & Bear, M. F. (2004). LTP and LTD: An embarrassment of riches. Neuron, 44, 5-21.
- Huganir, R. L., & Nicoll, R. A. (2013). AMPARs and synaptic plasticity: The last 25 years. Neuron, 80, 704-717.
