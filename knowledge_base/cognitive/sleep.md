---
title: Sleep
description: The recurring, reversible state of reduced responsiveness to the environment, its stages and regulation by circadian and homeostatic processes, proposed functions, and an Active Inference account of offline model optimisation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - sleep
  - circadian_rhythm
  - memory
semantic_relations:
  - type: relates
    links:
      - "[[rem_sleep]]"
      - "[[slow_wave_sleep]]"
      - "[[memory_consolidation]]"
      - "[[wakefulness]]"
      - "[[sleep_memory_ai]]"
wikidata: Q35831
---

# Sleep

## Overview
Sleep is a recurring, reversible state in which responsiveness to the environment is reduced and which is distinguished from coma or anaesthesia by its rapid reversibility and homeostatic regulation. Deprivation is followed by compensatory increases in sleep, and prolonged restriction impairs attention, memory and mood. Sleep is found across vertebrates and, in some form, in many invertebrates.

## Structure and regulation
- **Stages.** Human sleep is divided into non-REM stages, from light (N1, N2) to deep slow wave sleep (N3), and rapid eye movement sleep. A night comprises several cycles of roughly 90 minutes. Deep sleep dominates early cycles and [[rem_sleep]] later ones.
- **Two-process model.** Borbely (1982) proposed that sleep timing is governed by a homeostatic process, in which pressure to sleep accumulates during wake and dissipates in sleep, and a circadian process driven by the suprachiasmatic nucleus. Their interaction accounts for the usual timing and depth of sleep.
- **Proposed functions.** Leading hypotheses include consolidation of memory through offline reactivation (Diekelmann & Born, 2010), renormalisation of synaptic strength accumulated during waking (Tononi & Cirelli, 2014), and metabolic restoration. These are not mutually exclusive, and their relative importance is still debated.

## Relation to Active Inference
Hobson and Friston (2012) argued that sleep can be understood as a period in which the brain optimises its generative model without the cost of new sensory evidence. Because model evidence trades accuracy against complexity, redundant parameters accumulated during waking can be pruned offline, which has the same formal effect as synaptic down-scaling. Non-REM sleep is associated with this complexity reduction, while REM sleep is associated with exercising the model through internally generated activity. The proposal connects Active Inference with synaptic homeostasis and consolidation accounts, but it is a theoretical framework and its detailed predictions remain largely untested.

## Related concepts
- [[rem_sleep]] — the stage linked to dreaming and model exercise
- [[slow_wave_sleep]] — the deep stage linked to consolidation and synaptic renormalisation
- [[memory_consolidation]] — a central proposed function of sleep
- [[wakefulness]] — the state against which sleep pressure builds
- [[sleep_memory_ai]] — sleep-inspired memory processes in artificial systems
- [[synaptic_plasticity]] — the cellular substrate of sleep-dependent change

## References
- Borbely, A. A. (1982). A two process model of sleep regulation. Human Neurobiology, 1, 195-204.
- Diekelmann, S., & Born, J. (2010). The memory function of sleep. Nature Reviews Neuroscience, 11, 114-126.
- Tononi, G., & Cirelli, C. (2014). Sleep and the price of plasticity: From synaptic and cellular homeostasis to memory consolidation and integration. Neuron, 81, 12-34.
- Hobson, J. A., & Friston, K. J. (2012). Waking and dreaming consciousness: Neurobiological and functional considerations. Progress in Neurobiology, 98, 82-98.
