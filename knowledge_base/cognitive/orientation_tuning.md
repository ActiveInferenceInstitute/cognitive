---
title: Orientation Tuning
description: The selective response of neurons, first described in visual cortex, to edges or gratings at particular orientations, including tuning curves, cortical organisation and its place in models of neural coding.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - vision
  - neural_coding
  - visual_cortex
semantic_relations:
  - type: relates
    links:
      - "[[visual_cortex]]"
      - "[[neural_coding]]"
      - "[[neural_tuning]]"
      - "[[population_coding]]"
      - "[[predictive_coding]]"
---

# Orientation Tuning

## Overview
Orientation tuning is the property of a neuron to respond most strongly to a stimulus, usually a bar, edge or sinusoidal grating, at a particular orientation, with weaker responses as the stimulus rotates away from that preferred angle. It was described by Hubel and Wiesel (1962) in recordings from cat primary visual cortex and became the standard example of feature selectivity in sensory neurons.

## Tuning curves and organisation
A tuning curve plots firing rate against stimulus orientation. It is commonly summarised by a preferred orientation and a bandwidth, such as the half-width at half-height, and is often well described by a bell-shaped function of orientation. Orientation is periodic over 180 degrees for bars and gratings, so the angle is wrapped accordingly. Neurons differ in sharpness of tuning, and the width of the curve is relatively stable across changes in stimulus contrast, a property that constrains models of how tuning arises.

In many carnivores and primates, neurons with similar preferred orientation are grouped in columns, and preferred orientation changes smoothly across the cortical surface, interrupted by pinwheel points where all orientations meet. In rodents, neighbouring neurons have more varied preferences, so a columnar map is not universal.

## Origins
Hubel and Wiesel proposed a feedforward account in which a cortical cell receives input from several thalamic cells whose receptive fields are aligned along a line. Later work showed that intracortical excitation and inhibition also shape tuning, and the relative contribution of feedforward and recurrent circuits is still debated. Tuning is also modified by experience during development and by adaptation.

## Relation to Active Inference
The link is to predictive-processing models of vision, not to a derivation specific to this property. In predictive coding, units whose receptive fields resemble oriented, localised filters emerge when a generative model of natural images is fitted under a sparsity constraint (Olshausen and Field, 1996), and hierarchical predictive coding has been related to extra-classical effects such as end-stopping (Rao and Ballard, 1999). Tuned units can be read as encoding features that the model uses to explain input, with tuning width related to the precision with which a feature is represented. Such models are demonstrations of consistency with observed physiology and do not show that cortex actually implements them.

## Related concepts
- [[visual_cortex]] — the area where orientation tuning was discovered
- [[neural_tuning]] — selectivity of neurons for stimulus features in general
- [[neural_coding]] — how stimulus information is carried by neural activity
- [[population_coding]] — decoding stimulus orientation from many tuned neurons
- [[predictive_coding]] — a framework that explains oriented filters as learned features

## References
- Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. Journal of Physiology, 160, 106-154.
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381, 607-609.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: A functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience, 2, 79-87.
- Kandel, E. R., Koester, J. D., Mack, S. H., & Siegelbaum, S. A. (Eds.). (2021). Principles of Neural Science (6th ed.). McGraw-Hill.
