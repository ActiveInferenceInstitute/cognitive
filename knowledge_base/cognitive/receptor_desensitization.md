---
title: Receptor Desensitization
description: The reduction of a receptor's response during prolonged or repeated agonist exposure, covering ligand-gated channel desensitization, its first description at the motor end-plate, and its contribution to synaptic depression.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - receptors
  - synaptic_transmission
  - adaptation
semantic_relations:
  - type: relates
    links:
      - "[[synaptic_plasticity]]"
      - "[[receptor_phosphorylation]]"
      - "[[transmitter_release]]"
      - "[[neuromuscular_junction]]"
      - "[[sensory_adaptation]]"
---

# Receptor Desensitization

## Overview
Receptor desensitization is a decline in the response of a receptor to a continuing or repeated agonist, even though the agonist remains present. It is a general feature of ligand-gated ion channels and of G protein-coupled receptors, and it limits the duration and size of signals. It was first characterised by Katz and Thesleff (1957), who found that at the frog motor end-plate prolonged application of acetylcholine produced a response that declined despite continued agonist, and from which recovery took time after removal.

## Mechanism
For ligand-gated channels, the receptor cycles between closed resting, open and desensitized states. After agonist binds, the channel may open and then enter a desensitized conformation in which agonist remains bound but the pore is closed, and from which it returns to the resting state only slowly after the agonist unbinds. The rate and extent differ by receptor type: AMPA-type glutamate receptors desensitize within milliseconds, nicotinic acetylcholine receptors more slowly, and NMDA receptors in several ways. For G protein-coupled receptors, desensitization involves phosphorylation of the receptor by kinases and binding of arrestin proteins, which uncouples the receptor from its G protein (see [[receptor_phosphorylation]]). Desensitization should be distinguished from loss of receptors from the surface (down-regulation) and from depletion of transmitter at the presynaptic terminal (see [[transmitter_release]]).

## Synaptic significance
At some synapses, receptor desensitization shapes the response to trains of impulses. Where transmitter is released in large amounts or cleared slowly, as at some auditory synapses in the brainstem, desensitization of postsynaptic AMPA receptors contributes to short-term depression (Trussell et al., 1993); Jones and Westbrook (1996) review how receptor-specific desensitization kinetics can shape synaptic responses more generally. At others, the contribution is small compared with presynaptic depletion, so its importance depends on the synapse and the stimulation pattern. In cell-wide terms, desensitization is a fast form of adaptation, complementary to the slower processes in [[sensory_adaptation]].

## Relation to Active Inference
There is no direct link. Desensitization is a biophysical property and not a proposed computational mechanism, though its effects on gain could be incorporated in models of neuronal dynamics.

## Related concepts
- [[synaptic_plasticity]] — activity-dependent change of transmission
- [[receptor_phosphorylation]] — a regulatory mechanism for desensitization
- [[transmitter_release]] — presynaptic factors that also shape depression
- [[neuromuscular_junction]] — where it was first described
- [[post_tetanic]] — an opposing, enhancing short-term process

## References
- Katz, B., & Thesleff, S. (1957). A study of the 'desensitization' produced by acetylcholine at the motor end-plate. Journal of Physiology, 138, 63-80.
- Trussell, L. O., Zhang, S., & Raman, I. M. (1993). Desensitization of AMPA receptors upon multiquantal neurotransmitter release. Neuron, 10, 1185-1196.
- Jones, M. V., & Westbrook, G. L. (1996). The impact of receptor desensitization on fast synaptic transmission. Trends in Neurosciences, 19, 96-101.
