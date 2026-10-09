---
title: Emotion Recognition
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - emotion
  - social_cognition
  - perception
semantic_relations:
  - type: relates
    links:
      - "[[emotional_processing]]"
      - "[[social_cognition]]"
      - "[[perceptual_inference]]"
      - "[[active_inference]]"
---

# Emotion Recognition

## Overview
Emotion recognition is the inference of another individual's emotional state from observable cues: facial expressions, vocal prosody, body posture and movement, and context. It is a core component of [[social_cognition]], and it is also a task in affective computing, where machine systems classify emotion labels from face, voice, or physiological signals.

## Mechanism
Classic research following Ekman proposed a small set of basic emotions with universal facial signatures, and many experiments ask participants to match photographs of posed expressions to labels. This view has been challenged: reviews argue that facial configurations vary across cultures and situations, and that the mapping from a face to an emotion is far from one-to-one, so context, body, voice, and prior knowledge about the person strongly shape what is perceived. Constructionist theories hold that emotion categories are conceptual and learned.

Neural findings point to a distributed network: face-selective occipitotemporal cortex including the fusiform gyrus, the superior temporal sulcus for dynamic cues, the amygdala for salience and some threat-related cues, the insula, and orbitofrontal and [[medial_prefrontal]] regions for appraisal. Embodied accounts propose that recognition partly relies on simulating the observed state in one's own motor and visceral systems, which connects to [[imitation]]. Performance varies with development, age, and across clinical groups, and group differences are often small and task-dependent.

## Role in Active Inference
Recognition can be framed as perceptual inference: the observer holds a generative model in which hidden emotional states of another person cause expressive signals, and inverts it, weighting cues by their reliability ([[precision_weighting]], [[perceptual_inference]]). Context supplies priors, which explains why identical faces are read differently in different scenes, and combining face, voice, and body is an instance of [[multisensory_integration]]. When the observer's own emotional state is inferred through bodily signals, the same machinery applies to interoception. This framing is consistent with constructionist views, but Active Inference models specifically of emotion recognition are few and mostly theoretical; the connection is a conceptual fit, not an established result.

## Related concepts
- [[emotional_processing]] — the generation and representation of emotion
- [[theory_of_mind]] — inferring beliefs and intentions, versus feelings
- [[social_cognition]] — the broader domain
- [[multisensory_integration]] — combining face, voice, and posture
- [[emotion_regulation]] — altering one's own emotional responses
- [[autism]] — a condition in which recognition has been widely studied

## References
- Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). Emotional expressions reconsidered: challenges to inferring emotion from human facial movements. *Psychological Science in the Public Interest*.
- Picard, R. W. (1997). *Affective Computing*. MIT Press.
