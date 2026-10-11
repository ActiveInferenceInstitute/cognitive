---
title: Sparse Coding
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - neural-coding
  - efficient-coding
  - vision
semantic_relations:
  - type: relates
    links:
      - "[[neural_coding]]"
      - "[[predictive_coding]]"
      - "[[neural_efficiency]]"
aliases:
  - sparseness
---

# Sparse Coding

## Overview
Sparse coding is a representational strategy in which a stimulus is described by activating only a small fraction of the available units, drawn from a large dictionary of basis elements. In neuroscience it names the hypothesis that cortical populations represent natural stimuli with few active neurons at a time. In machine learning it is a method for learning such dictionaries.

## Mechanism
In Olshausen and Field's model, an image patch $I$ is approximated as a linear combination of basis functions $\phi_i$ with coefficients $a_i$. The coefficients and basis are chosen to minimize

$E = \sum_x \Big( I(x) - \sum_i a_i \phi_i(x) \Big)^2 + \lambda \sum_i S(a_i)$

where the first term penalizes reconstruction error and $S$ penalizes activity, for example with $S(a)=|a|$. The penalty corresponds to a heavy-tailed prior, such as a Laplace distribution, over coefficients, so the objective is a maximum a posteriori estimate under a linear generative model. Trained on natural image patches, the learned basis functions are localized, oriented and band-pass, resembling V1 simple-cell receptive fields.

The idea sits within the efficient coding tradition of Barlow, which proposes that sensory systems exploit redundancy in natural signals to represent them with fewer spikes or independent channels. Sparse codes are attractive because they allow high-capacity, low-energy representations and make associative learning easier. Experimental support includes sparse, selective firing in V1 under natural viewing, but measuring sparsity depends on assumptions about stimuli and units, and other models (independent components analysis, predictive coding with other priors) also reproduce oriented filters.

Inference in this model is iterative: coefficients are found by dynamics that combine feedforward drive with a feedback subtraction of the current reconstruction.

## Role in Active Inference
The structure above is a generative model with a prior and a likelihood, and the inference dynamics subtract top-down predictions from input, which is why sparse coding is often described as a close relative of [[predictive_coding]]. Rao and Ballard's hierarchical scheme can accommodate sparsity-inducing priors. Within [[active_inference]], sparsity is one choice of prior in a generative model, not a requirement of the framework. The sparse-coding result is evidence about the statistics of natural images and the form of V1 filters, not direct evidence for free-energy minimization.

## Related concepts
- [[neural_coding]] — wider family of coding hypotheses
- [[population_coding]] — distributed alternative representation
- [[feature_extraction]] — learned dictionaries as features
- [[visual_neuroscience]] — V1 receptive fields it explains
- [[predictive_coding]] — related inference scheme
- [[neural_efficiency]] — energy and capacity rationale

## References
- Barlow, H. B. (1961). Possible principles underlying the transformation of sensory messages. In Sensory Communication. MIT Press.
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature.
