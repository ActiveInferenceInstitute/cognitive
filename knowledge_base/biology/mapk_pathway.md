---
title: MAPK Pathway
description: The mitogen-activated protein kinase cascade that links extracellular signals to gene expression, its three-tier architecture, ultrasensitive dynamics, and roles in cell growth and in synaptic plasticity and memory.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - signalling
  - kinase
  - plasticity
semantic_relations:
  - type: relates
    links:
      - "[[camp_pathway]]"
      - "[[enzyme_cascades]]"
      - "[[synaptic_plasticity]]"
      - "[[gene_expression]]"
      - "[[transcription_factors]]"
      - "[[neural_plasticity]]"
---

# MAPK Pathway

## Overview
Mitogen-activated protein kinase (MAPK) pathways are conserved signalling modules that transmit information from the cell surface to the nucleus and other targets. In eukaryotes there are several parallel MAPK pathways, of which the extracellular signal-regulated kinase (ERK) pathway is the best known. It responds to growth factors and other stimuli and controls proliferation, differentiation, survival and other outcomes.

## Architecture
A canonical MAPK module has three tiers of kinases that activate each other in turn. A MAP kinase kinase kinase (MAPKKK, in the ERK pathway typically a Raf kinase) phosphorylates and activates a MAP kinase kinase (MAPKK, such as MEK), which phosphorylates a MAP kinase (such as ERK) on two sites, a threonine and a tyrosine. Activated ERK phosphorylates substrates in the cytoplasm and enters the nucleus, where it regulates transcription factors and thereby gene expression. The pathway is often initiated by receptor tyrosine kinases acting through the small GTPase Ras. Phosphatases reverse the phosphorylation, and scaffold proteins organise the components.

## Dynamics
Because activation needs two phosphorylations at each level of the cascade, the response can be ultrasensitive. Huang and Ferrell (1996) showed experimentally and in a model that the cascade responds in a switch-like way, steeper than a single enzyme reaction (see [[enzyme_cascades]]). Duration and strength of ERK activation can lead to different cell fates, so the same pathway can encode different outcomes through its temporal pattern.

## Role in synaptic plasticity
In neurons, activity and neurotrophin signals engage ERK, which is implicated in long-term potentiation and in some forms of long-term memory. Thomas and Huganir (2004) reviewed evidence that MAPK signalling regulates synaptic receptors, local protein synthesis and activity-dependent transcription. It often works with the cAMP pathway (see [[camp_pathway]]) and calcium signalling. Disruption of the pathway is also associated with developmental disorders and with cancer, in which mutations in Ras or Raf are common.

## Relation to Active Inference
There is no direct link to Active Inference. The pathway is part of the molecular machinery by which synapses change, and Active Inference treats learning at a more abstract level as updating of parameters of a generative model. How abstract learning rules map onto molecular cascades is an open question rather than a result.

## Related concepts
- [[camp_pathway]] — another second-messenger route in plasticity
- [[enzyme_cascades]] — the general principle of amplification in cascades
- [[synaptic_plasticity]] — the cellular process in which MAPK signalling takes part
- [[transcription_factors]] — nuclear targets of ERK
- [[memory_consolidation]] — long-term memory that depends on signalling to the nucleus

## References
- Pearson, G., et al. (2001). Mitogen-activated protein (MAP) kinase pathways: regulation and physiological functions. Endocrine Reviews, 22, 153-183.
- Huang, C.-Y. F., & Ferrell, J. E. (1996). Ultrasensitivity in the mitogen-activated protein kinase cascade. Proceedings of the National Academy of Sciences, 93, 10078-10083.
- Thomas, G. M., & Huganir, R. L. (2004). MAPK cascade signalling and synaptic plasticity. Nature Reviews Neuroscience, 5, 173-183.
