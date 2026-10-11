---
title: Brain Connectivity
description: Structural, functional and effective connectivity as distinct descriptions of how neural populations are linked, how each is estimated, and how effective connectivity relates to Active Inference.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neuroscience
  - networks
  - connectome
semantic_relations:
  - type: relates
    links:
      - "[[network_analysis]]"
      - "[[neural_synchrony]]"
      - "[[hierarchical_processing]]"
aliases:
  - connectivity_analysis
  - effective_connectivity
  - functional_connectivity
---

# Brain Connectivity

## Overview
Connectivity describes how neurons, populations and regions are linked and influence one another. Three notions are conventionally distinguished (Friston, 2011), and conflating them is a common source of error:
- **Structural connectivity**: anatomical links such as axonal projections and white-matter tracts, mapped by tracing in animals or diffusion imaging and tractography in humans. The set of all such links is the connectome (Sporns, Tononi and Kötter, 2005).
- **Functional connectivity**: statistical dependence between activity time series in different regions, such as correlation, coherence or mutual information. It is undirected and does not imply a causal or direct link.
- **Effective connectivity**: the directed influence one neural system exerts over another under a specified model of the interactions.

## Estimation
Functional connectivity is commonly computed from resting-state or task fMRI correlations, or from EEG and MEG phase and power coupling (see [[neural_synchrony]]). Directed measures such as Granger causality use temporal precedence, whereas dynamic causal modelling (Friston, Harrison and Penny, 2003) fits a generative model of neural dynamics and measurement and compares models of connectivity by their evidence. Graph-theoretic summaries, including modularity, hubs and path length, describe network organisation (see [[network_analysis]], [[network_science]]). Interpretation requires care: correlations can arise from common inputs, preprocessing choices alter estimates, and the structure-function relationship is imperfect.

## Role in Active Inference
Hierarchical predictive coding assigns connections a functional meaning: forward connections convey prediction errors, backward connections convey predictions, and lateral connections carry within-level interactions, with gain modulation reflecting precision (see [[predictive_coding]], [[precision_weighting]], [[bottom_up]]). Effective connectivity can therefore be estimated with dynamic causal modelling, which is itself built on variational inference, so the same free-energy machinery appears in both the theory of the brain and the analysis of its data. Parr, Friston and de Vries (2017) analysed how message passing on a graph relates to brain network structure. Whether estimated connection strengths correspond to the message-passing quantities in a particular model depends on the validity of that model, and many claimed correspondences remain proposals.

## Related concepts
- [[association_areas]] — hubs of integration
- [[consciousness_networks]] — large-scale networks relevant to awareness
- [[recurrent_networks]] — computational models with feedback connectivity
- [[message_passing]] — the algorithms that connections would implement

## References
- Friston, K. J., Harrison, L., & Penny, W. (2003). Dynamic causal modelling. NeuroImage, 19, 1273-1302.
- Sporns, O., Tononi, G., & Kötter, R. (2005). The human connectome: A structural description of the human brain. PLoS Computational Biology, 1, e42.
- Friston, K. J. (2011). Functional and effective connectivity: a review. Brain Connectivity, 1, 13-36.
- Parr, T., Friston, K., & de Vries, B. (2017). The graphical brain: belief propagation and active inference. Network Neuroscience, 1, 381-414.
