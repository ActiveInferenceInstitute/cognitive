---
title: Time Cells
description: Hippocampal and related neurons that fire at successive moments during a temporal interval, evidence from rodent delay tasks, relation to place cells, and Active Inference accounts of temporal structure.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - hippocampus
  - temporal_coding
  - episodic_memory
semantic_relations:
  - type: relates
    links:
      - "[[hippocampus]]"
      - "[[grid_cells]]"
      - "[[entorhinal_cortex]]"
      - "[[episodic_memory]]"
      - "[[subjective_time]]"
      - "[[temporal_coding]]"
---

# Time Cells

## Overview
Time cells are neurons that fire at specific moments within a temporally structured interval, so that a population of them tiles the interval as a sequence. They were first described in the rodent hippocampus, particularly area CA1, and are regarded as a temporal counterpart of place cells, which fire at specific locations. Each cell has a time field, a preferred latency after an event at which its firing peaks.

## Evidence
- **Delay periods.** Pastalkova and colleagues (2008) recorded from rats alternating between two goal arms while running in a wheel during a delay. Different CA1 neurons fired at different moments of the delay, producing a sequence that varied with the upcoming choice.
- **Memory for discontiguous events.** MacDonald and colleagues (2011) found that CA1 cells fired at particular moments during a delay between an odour and an object in a sequence memory task, which bridged the gap between events.
- **Time versus distance.** Kraus and colleagues (2013) separated elapsed time from distance travelled on a treadmill and found cells whose firing depended on one, the other, or both.
- **Wider distribution.** Cells with comparable properties have been reported in medial entorhinal cortex, prefrontal cortex and striatum, and human recordings have reported time-related responses.

Time fields tend to become wider later in the delay, so temporal resolution is coarser for more distant moments. Time cell sequences can change when task conditions change, which suggests that they depend on context and are not a fixed clock (Eichenbaum, 2014).

## Relation to Active Inference
Friston and Buzsáki (2016) discussed the functional anatomy of time and argued that the hippocampus and entorhinal cortex represent sequences of states, linking the coding of "what" and "when" to the temporal structure of a generative model. Sequential activity of time cells can be read as an implementation of such a sequence, and episodic memory and planning can both be described as traversing it. The link is a research programme, not a derivation: no result shows that time cells are required by Active Inference, and several other models explain time-field sequences.

## Related concepts
- [[hippocampus]] — the structure where time cells were first described
- [[grid_cells]] — the spatial counterpart in entorhinal cortex
- [[entorhinal_cortex]] — a second site of temporal and spatial coding
- [[episodic_memory]] — the memory type that depends on temporal context
- [[subjective_time]] — the experience of duration that the cells may support
- [[temporal_coding]] — the general principle of representing time in neural activity

## References
- Pastalkova, E., Itskov, V., Amarasingham, A., & Buzsáki, G. (2008). Internally generated cell assembly sequences in the rat hippocampus. Science, 321, 1322-1327.
- MacDonald, C. J., Lepage, K. Q., Eden, U. T., & Eichenbaum, H. (2011). Hippocampal "time cells" bridge the gap in memory for discontiguous events. Neuron, 71, 737-749.
- Eichenbaum, H. (2014). Time cells in the hippocampus: a new dimension for mapping memories. Nature Reviews Neuroscience, 15, 732-744.
- Friston, K., & Buzsáki, G. (2016). The functional anatomy of time: what and when in the brain. Trends in Cognitive Sciences, 20, 500-511.
- Kraus, B. J., Robinson, R. J., White, J. A., Eichenbaum, H., & Hasselmo, M. E. (2013). Hippocampal "time cells": Time versus path integration. Neuron, 78, 1090-1101.
