---
title: Auditory Processing
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - audition
  - sensory-processing
  - neuroscience
semantic_relations:
  - type: relates
    links:
      - "[[sensory_processing]]"
      - "[[predictive_processing]]"
      - "[[language_processing]]"
---

# Auditory Processing

## Overview
Auditory processing covers the transformation of sound pressure waves into neural signals and their interpretation as events, sources and speech. It includes peripheral encoding in the ear, subcortical analysis in the brainstem and midbrain, and cortical analysis that supports recognition, localization and scene organization.

## Mechanism
Sound moves the basilar membrane in the cochlea, whose mechanics decompose it by frequency: high frequencies peak near the base and low frequencies near the apex, a tonotopic map preserved up the pathway. Inner hair cells transduce motion into auditory nerve spikes, which phase-lock to the waveform at lower frequencies.

Ascending stages include the cochlear nucleus, the superior olivary complex, the inferior colliculus and the medial geniculate nucleus before primary auditory cortex (A1). Localization depends on interaural time and level differences compared in the superior olive, plus spectral cues from the outer ear for elevation. Cortical regions show selectivity for spectro-temporal patterns, pitch and, in higher areas, vocalizations.

A central problem is auditory scene analysis (Bregman): the ear receives a mixture, and the brain must group components into sources and segregate them over time (streaming). Cues include common onset, harmonicity, spatial location and continuity. Temporal structure matters throughout: auditory neurons encode envelope and fine structure on millisecond scales, and rhythm and speech perception depend on tracking slower modulations.

The mismatch negativity (MMN), an evoked response to a deviant sound in a repeated sequence, is a standard probe of auditory change detection (Näätänen).

## Role in Active Inference
Auditory sequences have strong temporal regularities, and the MMN is a frequently cited case for [[predictive_processing]]: a deviant violates a learned expectation and generates a prediction-error response, and the amplitude is modulated by the precision of the expectation. Friston and colleagues have modeled the MMN and related responses with hierarchical generative models. An alternative account attributes the MMN to stimulus-specific adaptation, i.e. fatigue of feature-tuned neurons, without any prediction, and the data from different paradigms do not cleanly separate the two.

## Related concepts
- [[sensory_processing]] — general principles of sensory systems
- [[speech_perception]] — the specialized perception of speech sounds
- [[language_processing]] — downstream linguistic interpretation
- [[neuroprosthetics]] — cochlear implants restoring peripheral encoding
- [[multisensory_integration]] — combination of sound with vision and touch
- [[temporal_binding]] — integration of events across time

## References
- Bregman, A. S. (1990). Auditory Scene Analysis: The Perceptual Organization of Sound. MIT Press.
