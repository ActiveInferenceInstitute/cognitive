---
title: Competitive Learning
description: An unsupervised learning scheme in which units compete for each input and the winner updates its weights, winner-take-all dynamics, self-organising maps, and relation to clustering and mixture models.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - unsupervised_learning
  - neural_networks
  - self_organization
semantic_relations:
  - type: relates
    links:
      - "[[unsupervised_learning]]"
      - "[[hebbian_learning]]"
      - "[[adaptive_resonance_theory]]"
      - "[[sparse_coding]]"
      - "[[expectation_maximization]]"
---

# Competitive Learning

## Overview
Competitive learning is a family of unsupervised learning rules in which units of a network compete to respond to each input, and only the winner, or a small neighbourhood around it, adapts. Through repeated presentations, each unit comes to represent a cluster of similar inputs, so the network discovers features and categories without a teacher. The principle was developed in the 1970s and 1980s by Grossberg, von der Malsburg, Rumelhart and Zipser, and Kohonen, among others.

## Mechanism
In the simplest form, units $j$ have weight vectors $\mathbf{w}_j$ and an input $\mathbf{x}$ is presented. The winner is the unit with the greatest activation, or equivalently, for normalised vectors, the one whose weights are closest to the input,
$$ j^{*}=\arg\min_j \lVert \mathbf{x}-\mathbf{w}_j\rVert,\qquad \Delta\mathbf{w}_{j^{*}}=\eta\,(\mathbf{x}-\mathbf{w}_{j^{*}}),$$
with learning rate $\eta$, while the other units are unchanged. The winner moves toward the input, so the weights approach the mean of the inputs they win. In networks, the competition is implemented by lateral inhibition among units, which gives winner-take-all dynamics (Rumelhart and Zipser, 1985; see [[mutual_inhibition]] and [[lateral_connections]]).

## Variants
- **Self-organising maps.** Kohonen (1982) added a neighbourhood function, so that units near the winner on a grid also update, which yields topology-preserving maps (see [[self_organization]]).
- **Adaptive resonance theory.** Grossberg's models add a vigilance test that creates a new category when no existing unit matches the input well enough, which addresses stability while learning (Grossberg, 1976; see [[adaptive_resonance_theory]] and [[stability_plasticity]]).
- **Neural and cortical links.** Competition and Hebbian strengthening have been used to explain orientation selectivity and topographic maps in sensory cortex (see [[hebbian_learning]] and [[map_plasticity]]).
Known problems include dead units that never win and sensitivity to initialisation.

## Relation to Active Inference
The link is a formal correspondence for the simplest case. Hard competitive learning with a decaying learning rate is an online form of k-means clustering. Soft competition, in which units share responsibility for each input in proportion to their posterior probability, corresponds to expectation maximisation in a mixture model (see [[expectation_maximization]] and [[latent_variable_models]]). Such learning can be read as inference about discrete hidden causes, consistent with generative model accounts, and with predictive coding and sparse coding models of cortical feature learning (see [[sparse_coding]] and [[predictive_coding]]). Active Inference itself adds action and expected free energy, which competitive learning does not include.

## Related concepts
- [[unsupervised_learning]] — learning structure without labels
- [[hebbian_learning]] — correlation-based synaptic change
- [[adaptive_resonance_theory]] — competitive learning with vigilance
- [[neural_computation]] — computing with networks of neurons
- [[pattern_recognition]] — a typical application

## References
- Rumelhart, D. E., & Zipser, D. (1985). Feature discovery by competitive learning. Cognitive Science, 9, 75-112.
- Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. Biological Cybernetics, 43, 59-69.
- Grossberg, S. (1976). Adaptive pattern classification and universal recoding: I. Parallel development and coding of neural feature detectors. Biological Cybernetics, 23, 121-134.
