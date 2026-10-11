---
title: Context Modulation
description: The adjustment of neural responses, processing parameters or preferences by contextual signals, from gain modulation and normalisation to context-dependent preferences in generative models.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - modulation
  - gain
  - preferences
semantic_relations:
  - type: relates
    links:
      - "[[gain_control]]"
      - "[[precision_weighting]]"
      - "[[preference_learning]]"
aliases:
  - contextual_modulation
---

# Context Modulation

## Overview
Context modulation is the scaling or reshaping of a system's response by signals that are not themselves the content being processed. In neuroscience it refers to how neural responses to a stimulus depend on surrounding stimuli, brain state or behavioural context. In modelling it extends to adjusting parameters, such as preferences, learning rates or the weight of evidence, according to the situation. This page is distinct from [[context_effects]], which catalogues behavioural phenomena.

## Mechanisms
- **Gain modulation**: a contextual variable multiplicatively scales responses without changing tuning. Salinas and Thier (2000) described this as a widespread computational principle, for example in parietal neurons whose responses to a visual target are scaled by eye or head position (see [[gain_control]]).
- **Divisive normalisation**: responses are divided by pooled activity of neighbouring neurons, producing surround suppression and contrast-dependent effects. Carandini and Heeger (2012) argued that normalisation is a canonical computation in many brain areas.
- **Neuromodulation**: acetylcholine, noradrenaline and dopamine alter excitability and plasticity according to state (see [[cholinergic_system]], [[dopamine_system]]).
- **Top-down feedback**: signals from higher areas change responses in earlier areas depending on task or attention (see [[attention_mechanisms]]).
- **Gating in learning models**: in adaptive resonance theory, a vigilance parameter and context signal determine whether input matches a stored category or recruits a new one (see [[adaptive_resonance_theory]]).

## Role in Active Inference
In hierarchical generative models, higher-level hidden states represent context and set parameters for lower levels, so that likelihoods, transitions or precisions depend on the inferred context (see [[hierarchical_inference]], [[precision_weighting]]). Gain modulation of prediction-error units is the proposed neural implementation of precision. Preferences are encoded in the prior over outcomes, and making these depend on a context state yields context-dependent preferences; the preference matrix can then be learned per context (see [[preference_learning]], [[observation_model]]). Such a modelling choice is straightforward in the discrete formulation, though preferences are often fixed in practice, and the empirical case for specific context-to-preference mappings in neural data is limited. Divisive normalisation and precision weighting have been related but are not identical, and their exact equivalence depends on the model.

## Related concepts
- [[predictive_processing]] — framework linking context and gain
- [[context_effects]] — behavioural consequences
- [[generative_model]] — structure in which context states are defined

## References
- Salinas, E., & Thier, P. (2000). Gain modulation: a major computational principle of the central nervous system. Neuron, 27, 15-21.
- Carandini, M., & Heeger, D. J. (2012). Normalization as a canonical neural computation. Nature Reviews Neuroscience, 13, 51-62.
