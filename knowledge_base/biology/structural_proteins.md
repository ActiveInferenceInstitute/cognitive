---
title: Structural Proteins
description: "Cytoskeletal, scaffolding and adhesion proteins that give neurons and synapses their shape, and how their remodelling underlies synaptic plasticity, axonal transport and myelin structure."
type: biology
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - molecular-biology
  - cytoskeleton
  - synapse
semantic_relations:
  - type: relates
    links:
      - "[[structural_plasticity]]"
      - "[[synaptic_plasticity]]"
      - "[[cell_biology]]"
---

# Structural Proteins

## Overview
Structural proteins are the proteins that provide physical form, scaffolding and mechanical stability to cells and the specialized junctions between them. In neuroscience the term covers the cytoskeleton, the scaffolds that organize synapses, adhesion molecules, and proteins of the myelin sheath. They are the molecular building blocks that make the changes described in [[structural_plasticity]] possible, but they are not themselves a computational mechanism.

## Classes
The cytoskeleton has three main components. Actin filaments are concentrated in dendritic spines, where polymerization and depolymerization, regulated by proteins such as cofilin, determine spine shape and size. Microtubules, polymers of tubulin, form tracks along axons and dendrites on which the motor proteins kinesin and dynein carry cargo (vesicles, mitochondria, mRNA); microtubule-associated proteins such as tau stabilize them. Neurofilaments give axons tensile strength and set caliber, which influences conduction speed.

At the synapse, scaffold proteins of the postsynaptic density, including PSD-95, Homer and Shank, anchor glutamate receptors and signalling enzymes near the active zone and link them to the actin cytoskeleton. Cell adhesion molecules such as neurexins and neuroligins, cadherins and integrins span the synaptic cleft and hold pre- and postsynaptic sides in alignment while contributing to synapse formation. In myelin, myelin basic protein and proteolipid protein compact membrane layers ([[myelination]]).

Plasticity depends on remodelling these structures. After induction of [[ltp]], actin polymerization enlarges the spine and PSD scaffolds recruit more receptors. The synaptic tagging and capture hypothesis (Frey and Morris, 1997) proposes that a tagged synapse can capture plasticity-related proteins made elsewhere in the neuron, so that structural proteins, among other products of [[gene_expression]] and [[protein_synthesis]], are delivered to the sites that need them. This links short-lived signals to durable structure ([[memory_consolidation]], [[synaptic_consolidation]]). Mutations in scaffold and adhesion genes have been associated with neurodevelopmental conditions in genetic studies, a topic of ongoing research.

## Role in Active Inference
There is no direct connection. Active Inference operates at the level of computation, and structural proteins constitute part of the implementation substrate on which proposed learning rules would act. They matter to the theory only insofar as neural-process theories require that synaptic efficacy change in a way consistent with the proposed updates ([[hebbian_learning]]); this is an implementation question, not a theoretical claim.

## Related concepts
- [[molecular_biology]] — wider context of protein structure and function
- [[neural_plasticity]] — functional consequences of structural change
- [[memory_systems]] — systems role for synaptic structure
- [[neural_development]] — scaffold and adhesion roles in synapse formation

## References
- Frey, U., & Morris, R. G. M. (1997). Synaptic tagging and long-term potentiation. *Nature*, 385.
