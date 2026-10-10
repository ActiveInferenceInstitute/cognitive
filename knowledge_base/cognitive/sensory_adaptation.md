---
title: Sensory Adaptation
description: "The reduction of sensory responsiveness under sustained or repeated stimulation, its mechanisms, its distinction from habituation, and predictive-coding accounts of repetition suppression."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - perception
  - plasticity
  - sensory-systems
semantic_relations:
  - type: relates
    links:
      - "[[sensory_processing]]"
      - "[[gain_control]]"
      - "[[neural_plasticity]]"
      - "[[prediction_error]]"
---

# Sensory Adaptation

## Overview
Sensory adaptation is the change in a sensory system's response after prolonged or repeated stimulation, typically a decline in sensitivity to the persistent stimulus together with increased relative sensitivity to changes around it. Familiar cases include dark and light adaptation in vision, the fading of an ambient smell, and the motion aftereffect after watching a waterfall.

## Mechanisms and distinctions
- **Peripheral.** Photoreceptors and other receptors adjust their gain through intracellular signalling, so the dynamic range tracks the ambient level (light adaptation).
- **Cortical.** Neurons in visual and auditory cortex reduce responses to a repeated feature and often shift their tuning, producing aftereffects such as tilt and motion aftereffects.
- **Timescales.** Adaptation operates from tens of milliseconds to minutes and longer, and several timescales can coexist in one system.
- **Versus habituation.** Habituation is a learned decrement in a behavioural response to a repeated harmless stimulus, recoverable by a novel stimulus (dishabituation). Adaptation is a stimulus-specific change in sensory processing that is not reducible to receptor fatigue.
- **Functional accounts.** Adaptation is commonly interpreted as gain control that matches the limited response range of neurons to the current stimulus statistics, in the spirit of efficient coding ([[gain_control]], [[sparse_coding]]).

## Role in Active Inference
Predictive coding interprets repetition suppression as a reduction in prediction error: a repeated stimulus becomes predicted, so error units fire less. Models by Auksztulewicz and Friston use this to explain how expectation and repetition interact. The account is contested, since adaptation-based explanations that do not invoke expectation also account for much of the data, and experiments that manipulate expectation separately from repetition give mixed results. Treat predictive coding here as one candidate explanation among several.

## Related concepts
- [[gain_control]] - the normalisation mechanism most associated with adaptation
- [[prediction_error]] - the quantity said to be reduced under repetition suppression
- [[neural_plasticity]] - adaptation as short-term plasticity
- [[homeostatic_plasticity]] - slower regulation of activity levels
- [[perceptual_learning]] - longer-lasting, practice-driven change

## References
- Wark, B., Lundstrom, B. N., & Fairhall, A. (2007). Sensory adaptation. Current Opinion in Neurobiology.
- Grill-Spector, K., Henson, R., & Martin, A. (2006). Repetition and the brain: neural models of stimulus-specific effects. Trends in Cognitive Sciences.
- Auksztulewicz, R., & Friston, K. (2016). Repetition suppression and its contextual determinants in predictive coding. Cortex.
