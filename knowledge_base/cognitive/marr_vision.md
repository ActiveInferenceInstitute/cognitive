---
title: Marr's Theory of Vision
description: David Marr's computational theory of vision, built on three levels of analysis and a sequence of representations from the primal sketch to three-dimensional models, and its relation to Bayesian approaches.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - vision
  - levels_of_analysis
  - computational_neuroscience
semantic_relations:
  - type: relates
    links:
      - "[[computational_vision]]"
      - "[[edge_detection]]"
      - "[[depth_perception]]"
      - "[[object_recognition]]"
      - "[[visual_perception]]"
---

# Marr's Theory of Vision

## Overview
David Marr's book Vision (1982) presented a computational account of how a visual system could recover a description of the three-dimensional world from two-dimensional retinal images. It is as influential for its methodology as for its specific proposals. Marr argued that an information-processing system must be understood at three levels, and that neglecting the first leads to explanations that describe mechanisms without saying what they accomplish.

## Three levels of analysis
- **Computational theory.** What problem is solved, why, and what constraints from the world make it solvable.
- **Representation and algorithm.** How inputs and outputs are represented and which procedure transforms one into the other.
- **Hardware implementation.** How the representation and algorithm are physically realised, for example in neurons.

## Stages of representation
Marr proposed a bottom-up sequence of increasingly abstract descriptions:
1. **Primal sketch.** Intensity changes, edges, bars and blobs, found for example from zero-crossings of a filtered image (Marr and Hildreth, 1980; see [[edge_detection]]).
2. **2.5D sketch.** A viewer-centred map of surface orientation and depth derived from cues such as stereo disparity, shading and motion (see [[depth_perception]]).
3. **3D model representation.** An object-centred description, built from hierarchical arrangements of generalised cylinders, supporting recognition (see [[object_recognition]]).

## Influence and critique
The levels framework is still widely cited, and the algorithmic work launched modern [[computational_vision]]. Critics note that the strictly feedforward, modular pipeline gives little role to attention, top-down knowledge and task, and that physiological findings on extensive feedback do not fit it well. Many modern models also blur the separation between levels.

## Relation to Active Inference
Marr's levels are often used to locate the Free Energy Principle: it is offered as a computational-level principle, with variational inference and predictive coding as algorithmic proposals. The Bayesian approach reinterprets vision as inverting a generative model of image formation, in which top-down predictions play a central role, differing from Marr's bottom-up pipeline. The relationship is therefore one of reframing and partial contrast rather than direct derivation.

## Related concepts
- [[computational_vision]] — the field Marr's programme helped define
- [[edge_detection]] — the first stage of the primal sketch
- [[depth_perception]] — recovery of surface layout
- [[object_recognition]] — the aim of the final representation
- [[predictive_coding]] — a contrasting, feedback-centred account

## References
- Marr, D. (1982). Vision: A Computational Investigation into the Human Representation and Processing of Visual Information. W. H. Freeman.
- Marr, D., & Hildreth, E. (1980). Theory of edge detection. Proceedings of the Royal Society of London B, 207, 187-217.
