---
title: Temporal Processing
description: "How the nervous system represents time, covering order, duration, rhythm, and synchrony across millisecond to circadian scales, timing models, and hierarchies of timescales in generative models."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - time
  - perception
  - timing
semantic_relations:
  - type: relates
    links:
      - "[[temporal_integration]]"
      - "[[temporal_attention]]"
      - "[[temporal_models]]"
      - "[[sequence_learning]]"
---

# Temporal Processing

## Overview
Temporal processing is the sensing, representation, and use of time by the brain: judging which of two events came first, estimating how long something lasted, perceiving rhythm, and predicting when something will happen. Time is not sensed by a dedicated receptor, so it is inferred from neural dynamics and from the structure of events.

## Scales
- **Milliseconds.** Temporal order and simultaneity judgments, gap detection, and temporal resolution in vision ([[contrast_sensitivity]], [[apparent_motion]]) and audition. Sensory latencies differ across modalities, so perceived simultaneity is not the same as physical simultaneity, and the system can recalibrate to a consistent lag.
- **Hundreds of milliseconds to seconds.** Interval timing, rhythm, and beat perception, and the dynamics of attention ([[attentional_blink]], [[temporal_attention]]).
- **Longer.** Interval estimates over minutes, sleep-wake and circadian cycles, and the ordering of memories ([[episodic_memory]]).

## Timing models
- **Scalar timing.** Interval timing in animals and humans shows the scalar property: variability grows in proportion to the timed duration (Weber-like). Scalar expectancy theory explains this with a pacemaker, accumulator, and memory comparison.
- **State-dependent networks.** An alternative holds that time is read out from the evolving state of a neural network, without a dedicated clock.
- **Oscillatory and ramping accounts.** Rhythmic neural activity and ramping firing rates have each been proposed as timing signals.
- **Brain regions.** Basal ganglia, cerebellum, supplementary motor area, and prefrontal cortex are recurrently implicated, with different contributions for sub-second and supra-second timing.

None of these accounts is settled, and timing may rely on different mechanisms at different scales.

## Role in Active Inference
Generative models in Active Inference need an account of time. Hierarchical models assign slower dynamics to higher levels and faster dynamics to lower ones, so that a sequence at one level provides context for events at the next ([[temporal_models]]). Perceived timing can then be treated as inference about when events occurred and will occur, and precision weighting over time corresponds to [[temporal_attention]]. This offers a formalism; empirical support for particular timing circuits is limited.

## Related concepts
- [[temporal_integration]] - accumulation over time windows
- [[temporal_coding]] - spike timing as information
- [[sequence_learning]] - learning ordered structure
- [[temporal_discounting]] - the valuation of delayed outcomes
- [[temporal_binding]] - unifying events close in time

## References
- Gibbon, J. (1977). Scalar expectancy theory and Weber's law in animal timing. Psychological Review.
- Kiebel, S. J., Daunizeau, J., & Friston, K. J. (2008). A hierarchy of time-scales and the brain. PLoS Computational Biology.
