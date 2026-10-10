---
title: fMRI Studies
description: Functional magnetic resonance imaging studies infer brain activity from blood-oxygen-level-dependent signals, offering millimetre-scale spatial maps with slow temporal resolution and well-known inferential pitfalls.
type: research
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neuroimaging
  - methods
  - bold
semantic_relations:
  - type: relates
    links:
      - "[[eeg_recording]]"
      - "[[cognitive_neuroscience]]"
      - "[[predictive_processing]]"
      - "[[neural_decoding]]"
---

# fMRI Studies

## Overview
Functional magnetic resonance imaging (fMRI) studies measure the blood-oxygen-level-dependent (BOLD) signal, an indirect index of neural activity, while participants perform tasks or rest. They are the most widely used method for mapping the human brain regions associated with cognitive functions.

## Methods and design
**Signal.** Neural activity changes local blood flow and oxygenation, altering the magnetic properties of tissue (Ogawa et al., 1990). The haemodynamic response rises over a few seconds and peaks around 4 to 6 s after a brief event, so temporal resolution is on the order of seconds, in contrast to the millisecond resolution of EEG. Spatial resolution is typically a few millimetres. The signal reflects synaptic input and local processing more than spiking output, and its relation to neural activity is not strictly linear.

**Designs.** Block designs alternate sustained conditions and have high statistical power; event-related designs present brief trials and allow estimation of responses to individual events. Resting-state scans analyse spontaneous fluctuations, yielding functional connectivity networks such as the default mode network.

**Analysis.** Mass-univariate analysis fits a general linear model at each voxel. Multivoxel pattern analysis decodes stimulus or state information from distributed patterns. Connectivity methods include psychophysiological interactions and dynamic causal modelling (Friston, Harrison and Penny, 2003).

**Pitfalls.**
- Reverse inference: activation of a region does not show that a particular cognitive process occurred, since regions participate in many functions (Poldrack, 2006).
- Circular analysis can produce inflated brain-behaviour correlations (Vul et al., 2009).
- Cluster-wise correction methods used in some software showed inflated false-positive rates (Eklund, Nichols and Knutsson, 2016).
- Small samples and head motion limit reliability.

## Role in Active Inference
fMRI is used to test predictive-processing hypotheses and to fit generative models of neural dynamics. Studies report reduced sensory-cortex activity for predicted stimuli, for example reduced primary visual cortex activity when local features are organised into a coherent shape (Murray et al., 2002), interpreted as explaining away by top-down predictions. Dynamic causal modelling provides a Bayesian framework for inferring effective connectivity from BOLD data. Because the BOLD signal cannot separate predictions from prediction errors and has limited temporal resolution, such findings are compatible with the theory but do not discriminate it from alternatives.

## Related concepts
- [[eeg_recording]] — complementary temporal resolution
- [[default_mode]] — a network characterised largely through resting-state fMRI
- [[frontoparietal]] — control networks mapped by fMRI
- [[hyperscanning]] — joint recording of interacting participants
- [[neuroprosthetics]] — decoding applications related to pattern analysis

## References
- Ogawa, S., et al. (1990). Brain magnetic resonance imaging with contrast dependent on blood oxygenation. Proceedings of the National Academy of Sciences.
- Poldrack, R. A. (2006). Can cognitive processes be inferred from neuroimaging data? Trends in Cognitive Sciences.
- Eklund, A., Nichols, T. E., & Knutsson, H. (2016). Cluster failure: why fMRI inferences for spatial extent have inflated false-positive rates. Proceedings of the National Academy of Sciences.
- Friston, K. J., Harrison, L., & Penny, W. (2003). Dynamic causal modelling. NeuroImage.
- Vul, E., Harris, C., Winkielman, P., & Pashler, H. (2009). Puzzlingly high correlations in fMRI studies of emotion, personality, and social cognition. Perspectives on Psychological Science.
- Murray, S. O., Kersten, D., Olshausen, B. A., Schrater, P., & Woods, D. L. (2002). Shape perception reduces activity in human primary visual cortex. Proceedings of the National Academy of Sciences.
