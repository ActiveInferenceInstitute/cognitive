---
title: Network Neuroscience
description: The study of the brain as a network of interacting regions and cells using graph theory, covering structural, functional and effective connectivity, hubs and modules, and its ties to Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - connectomics
  - graph_theory
  - brain_networks
semantic_relations:
  - type: relates
    links:
      - "[[connectivity]]"
      - "[[small_world]]"
      - "[[graph_theory]]"
      - "[[attention_networks]]"
      - "[[default_mode]]"
---

# Network Neuroscience

## Overview
Network neuroscience treats the brain as a network, with nodes that are neurons, populations or regions and edges that are physical connections or statistical dependencies. Its aim is to relate the organisation of these connections to function, development, ageing and disease. The approach grew with large-scale imaging, tract tracing and electrophysiology, which provide connectivity data, and with network science, which provides measures for analysing them.

## Types of connectivity
- **Structural connectivity** describes anatomical links, such as axonal projections or white matter tracts estimated with diffusion imaging.
- **Functional connectivity** describes statistical dependence between activity time series, for example correlation of fMRI signals, without implying direction or a causal route.
- **Effective connectivity** describes the directed influence of one neural system over another under a model of the interaction. Dynamic causal modelling is a model-based method of this kind.

## Graph measures
Typical measures include degree, clustering, characteristic path length, modularity and centrality. Brain networks are reported to be modular, to have a [[small_world]] organisation, and to contain hubs, some of which are densely interconnected in a rich club. Modules often correspond to known systems, such as the default mode and attention networks. Results depend on choices of parcellation, thresholding and data quality, and functional connectivity can differ from the underlying anatomy, so conclusions need to be treated with care.

## Relation to Active Inference
The link is practical and in places direct. Friston distinguished functional from effective connectivity and developed dynamic causal modelling to infer the latter, which fits a generative model of neural dynamics to data by variational Bayesian inference. In predictive-coding schemes the hierarchical architecture of the cortex, with forward and backward connections, is read as message passing for inference, so network structure is linked to the computation performed. Whether the brain's measured network topology is explained by free energy minimisation is a research question and not a settled result.

## Related concepts
- [[connectivity]] — structural, functional and effective links between regions
- [[small_world]] — a characteristic topology of brain graphs
- [[graph_theory]] — the mathematical toolkit for network measures
- [[attention_networks]] — large-scale systems identified in network analyses
- [[default_mode]] — a major resting-state network
- [[scale_free_networks]] — heavy-tailed degrees and hubs

## References
- Bullmore, E., & Sporns, O. (2009). Complex brain networks: graph theoretical analysis of structural and functional systems. Nature Reviews Neuroscience, 10, 186-198.
- Sporns, O. (2011). Networks of the Brain. MIT Press.
- Friston, K. J. (2011). Functional and effective connectivity: a review. Brain Connectivity, 1, 13-36.
