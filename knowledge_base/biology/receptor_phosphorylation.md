---
title: Receptor Phosphorylation
description: Covalent addition of phosphate groups to receptor proteins by kinases, how it changes receptor activity, trafficking and signalling, and its role in synaptic plasticity.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - cell_signalling
  - post_translational_modification
  - synaptic_receptors
semantic_relations:
  - type: relates
    links:
      - "[[second_messengers]]"
      - "[[nmda_receptors]]"
      - "[[synaptic_plasticity]]"
      - "[[protein_trafficking]]"
      - "[[neural_plasticity]]"
---

# Receptor Phosphorylation

## Overview
Receptor phosphorylation is the covalent attachment of a phosphate group, usually to a serine, threonine or tyrosine side chain, on a receptor protein. Protein kinases add the phosphate and protein phosphatases remove it, so the modification is reversible and fast. Hunter (1995) described kinases and phosphatases as opposing partners that together set the phosphorylation state of signalling proteins.

## Effects on receptors
Phosphorylation adds negative charge and can change a receptor's shape, so it may act in several ways.
- **Activity.** It can raise or lower channel conductance or open probability, as in many ligand-gated ion channels.
- **Sensitivity.** In G protein-coupled receptors, phosphorylation of the activated receptor promotes binding of arrestins, which uncouple it from G proteins and favour internalisation, a process called desensitisation.
- **Signalling.** Receptor tyrosine kinases autophosphorylate on ligand binding, creating docking sites for adaptor proteins that start downstream cascades.
- **Trafficking.** Phosphorylation can alter receptor internalisation, recycling and surface delivery, linking it to [[protein_trafficking]].

Kinases are themselves regulated by [[second_messengers]] such as cyclic AMP and calcium, so receptor phosphorylation is a node where several signalling pathways meet. The human genome encodes several hundred protein kinases (Manning et al., 2002).

## Role in synaptic plasticity
At excitatory synapses, phosphorylation of glutamate receptor subunits and their partners by kinases such as protein kinase A and calcium/calmodulin-dependent kinase II is implicated in changes of synaptic strength that follow patterned activity (see [[synaptic_plasticity]], [[ltp]] and [[nmda_receptors]]). The details differ by receptor subtype, brain region and induction protocol, and the relative importance of channel modification and receptor trafficking is still debated.

## Relation to Active Inference
The connection is a matter of implementation rather than theory. Active Inference models learning as updating of parameters, and phosphorylation-dependent changes in synaptic efficacy are one candidate biological mechanism for that updating. Mapping a particular modification onto a particular model quantity is an interpretation that has not been established.

## Related concepts
- [[second_messengers]] — signals that activate the relevant kinases
- [[nmda_receptors]] — receptors whose phosphorylation shapes plasticity
- [[ltp]] — the plasticity phenomenon most studied in this context
- [[protein_trafficking]] — delivery and removal of phosphorylated receptors
- [[transmitter_release]] — the presynaptic counterpart in synaptic modulation

## References
- Hunter, T. (1995). Protein kinases and phosphatases: the yin and yang of protein phosphorylation and signaling. Cell, 80, 225-236.
- Manning, G., et al. (2002). The protein kinase complement of the human genome. Science, 298, 1912-1934.
- Alberts, B., et al. (2015). Molecular Biology of the Cell (6th ed.). Garland Science.
