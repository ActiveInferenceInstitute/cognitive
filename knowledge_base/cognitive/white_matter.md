---
title: White Matter
description: The myelinated axonal fibre tracts that connect brain regions, their types, measurement with diffusion MRI, development and plasticity, and how structural connectivity constrains message passing models of the brain.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neuroanatomy
  - myelin
  - connectivity
semantic_relations:
  - type: relates
    links:
      - "[[myelination]]"
      - "[[connectivity]]"
      - "[[neural_circuits]]"
      - "[[neural_development]]"
      - "[[neural_plasticity]]"
---

# White Matter

## Overview
White matter is the part of the central nervous system composed mainly of myelinated axons, which gives it a pale appearance in fresh tissue. It forms the long-range wiring between grey-matter regions, which contain neuronal cell bodies and dendrites. Myelin, produced by oligodendrocytes, insulates axons and permits fast saltatory conduction, so that signals can travel between distant regions with timing precise enough for coordinated computation. White matter accounts for a large fraction of the volume of the human brain.

## Organisation and measurement
- **Tract classes.** Fibres are grouped into association tracts, which link regions within a hemisphere (such as the arcuate fasciculus and the superior longitudinal fasciculus), commissural tracts, which link the hemispheres (chiefly the corpus callosum), and projection tracts, which connect cortex with the thalamus, brainstem and spinal cord (Catani & Thiebaut de Schotten, 2012).
- **Diffusion MRI.** Water diffuses preferentially along axons, so diffusion tensor imaging (Basser et al., 1994) can estimate fibre orientation and, with tractography, reconstruct pathways. Indices such as fractional anisotropy are sensitive to fibre organisation but not specific to myelin, and tractography can generate false connections, so it provides estimates rather than direct anatomy.
- **Development and plasticity.** Myelination proceeds through childhood and adolescence, and continues in some frontal and association regions into adulthood. Learning and experience are associated with changes in white matter structure, and neural activity can influence myelination, suggesting that it contributes to plasticity as well as to hardwired transmission (Fields, 2008).
- **Disease.** Damage or abnormal development is implicated in multiple sclerosis, which involves demyelination, in leukodystrophies, in traumatic injury, and in a range of psychiatric and developmental conditions, though findings in the latter are often small and not specific.

## Relation to Active Inference
White matter supplies the anatomical substrate for message passing between levels of a hierarchical model. Structural connectivity determines which regions can exchange predictions and prediction errors, and conduction delays shape the timing of that exchange. Dynamic causal modelling uses such constraints to infer effective connectivity from data. The connection is architectural and supportive; it is not a theory of what white matter computes.

## Related concepts
- [[myelination]] — the process that creates the insulating sheaths
- [[connectivity]] — structural and functional links between regions
- [[neural_circuits]] — assemblies of neurons linked by these fibres
- [[neural_development]] — maturation of tracts across the lifespan
- [[neural_plasticity]] — experience-dependent change that includes myelin
- [[cognitive_development_expanded]] — developmental changes in cognition

## References
- Basser, P. J., Mattiello, J., & LeBihan, D. (1994). MR diffusion tensor spectroscopy and imaging. Biophysical Journal, 66, 259-267.
- Fields, R. D. (2008). White matter in learning, cognition and psychiatric disorders. Trends in Neurosciences, 31, 361-370.
- Catani, M., & Thiebaut de Schotten, M. (2012). Atlas of Human Brain Connections. Oxford: Oxford University Press.
