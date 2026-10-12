---
title: Neural Tuning
description: The dependence of a neuron's response on a stimulus or movement parameter, described by tuning curves, with classic examples, the information carried by tuned populations, and links to probabilistic coding.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neural_coding
  - tuning_curves
  - population_coding
semantic_relations:
  - type: relates
    links:
      - "[[neural_coding]]"
      - "[[population_coding]]"
      - "[[orientation_tuning]]"
      - "[[firing_rate]]"
      - "[[fisher_information]]"
---

# Neural Tuning

## Overview
Neural tuning describes how the response of a neuron varies with a stimulus or behavioural variable. The relation is summarised in a tuning curve, which plots mean firing rate (see [[firing_rate]]) against a parameter such as stimulus orientation, motion direction, sound frequency, spatial position or movement direction. The value that evokes the strongest response is the preferred value, and the width of the curve describes how selective the neuron is. The notion underlies much of [[neural_coding]].

## Classic examples
- **Orientation tuning.** Neurons in primary visual cortex respond most to bars of a particular orientation and less to others (Hubel and Wiesel, 1962; see [[orientation_tuning]] and [[hubel_wiesel]]).
- **Direction tuning in motor cortex.** Neurons in primary motor cortex fire most when a monkey reaches in a preferred direction, with a broad, roughly cosine-shaped dependence on the angle (Georgopoulos and colleagues, 1982).
- **Place and frequency.** Hippocampal place cells are tuned to location and auditory neurons to sound frequency.

Tuning is not fixed. It can change with attention, adaptation, learning and behavioural context.

## Tuning and information
A single tuned neuron is ambiguous, since a given rate can come from several stimuli. Reading out a stimulus therefore usually uses a population of neurons with different preferred values (see [[population_coding]] and [[population_vector]]). For independent Poisson neurons with tuning curves $f_i(\theta)$, the Fisher information about the stimulus $\theta$ is $J(\theta)=\sum_i f_i'(\theta)^2 / f_i(\theta)$, so information is highest where the curves are steep and not at their peaks (see [[fisher_information]]). Correlated variability among neurons modifies this result.

## Relation to Active Inference
Tuning curves can be viewed as the likelihood function of a neuron's response given a stimulus. In probabilistic population code theory, population activity encodes a probability distribution over the stimulus, and Poisson-like variability allows evidence from different sources to be combined by adding activity (Ma and colleagues, 2006). Predictive coding theories further treat tuning as the product of learning a generative model with sparse, efficient features (see [[sparse_coding]]). These are theories of coding, with experimental support in some systems and debate in others.

## Related concepts
- [[orientation_tuning]] — tuning of visual cortical neurons to edge angle
- [[population_coding]] — joint representation across tuned cells
- [[population_vector]] — a readout method from tuned populations
- [[neural_coding]] — the wider question of how neurons carry information
- [[sparse_coding]] — an efficient-coding account of why tuning emerges

## References
- Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. Journal of Physiology, 160, 106-154.
- Georgopoulos, A. P., Kalaska, J. F., Caminiti, R., & Massey, J. T. (1982). On the relations between the direction of two-dimensional arm movements and cell discharge in primate motor cortex. Journal of Neuroscience, 2, 1527-1537.
- Ma, W. J., Beck, J. M., Latham, P. E., & Pouget, A. (2006). Bayesian inference with probabilistic population codes. Nature Neuroscience, 9, 1432-1438.
