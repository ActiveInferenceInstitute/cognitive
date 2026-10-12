---
title: Cocktail Party Effect
description: The ability to follow one speaker among competing voices, the cues and attentional mechanisms involved, and its treatment as precision-weighted inference about auditory sources.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - auditory_attention
  - speech_perception
  - selective_attention
semantic_relations:
  - type: relates
    links:
      - "[[selective_attention]]"
      - "[[auditory_processing]]"
      - "[[speech_perception]]"
      - "[[attention_mechanisms]]"
      - "[[precision_weighting]]"
wikidata: Q673205
---

# Cocktail Party Effect

## Overview
The cocktail party problem is the task of following one speaker in the presence of other voices and background noise. Cherry (1953) named it and studied it with dichotic listening, in which a different message is played to each ear and the listener repeats one aloud. Listeners succeed easily at this, retaining the physical features of the unattended message, such as a change of voice from male to female, but very little of its content.

## Cues and models
Segregating a voice from a mixture relies on spatial location, differences in pitch and timbre, onset asynchrony, and continuity over time, a set of processes grouped under auditory scene analysis (Bregman, 1990). Seeing the speaker's lips and knowing the language and topic help further. Some unattended material can nevertheless break through: Moray (1959) found that listeners often noticed their own name in the ignored channel. Early theories differed over where selection occurs. Broadbent's filter model placed it early, based on physical features, while Treisman's attenuation model allowed the unattended signal to be weakened rather than blocked, so that highly relevant words could still be detected.

## Neural findings
Recordings in human auditory cortex show that responses in higher auditory areas track the speech of the attended talker more strongly than that of the ignored talker, and that the attended voice can be decoded from brain activity in multi-speaker mixtures (Mesgarani and Chang, 2012). Such tracking is thought to reflect top-down modulation by attention on top of bottom-up segregation.

## Relation to Active Inference
In predictive-processing terms, listening in a mixture is inference about hidden sources that jointly produced the sound, with attention implemented as increasing the precision, or gain, of prediction errors associated with the chosen stream (Feldman and Friston, 2010, treat attention in this way for vision). Generative models of speech supply the strong priors that make the target predictable. This is a conceptual mapping with some modelling support rather than a quantified account of the phenomenon, and established psychoacoustic models describe much of the same behaviour without it.

## Related concepts
- [[selective_attention]] — the general capacity to prioritise one input
- [[auditory_processing]] — the sensory stages that precede selection
- [[speech_perception]] — recognition of the attended message
- [[attention_mechanisms]] — theories of filtering and gain control
- [[precision_weighting]] — the Active Inference expression of attentional gain

## References
- Cherry, E. C. (1953). Some experiments on the recognition of speech, with one and with two ears. Journal of the Acoustical Society of America, 25, 975-979.
- Moray, N. (1959). Attention in dichotic listening: affective cues and the influence of instructions. Quarterly Journal of Experimental Psychology, 11, 56-60.
- Bregman, A. S. (1990). Auditory Scene Analysis: The Perceptual Organization of Sound. MIT Press.
- Mesgarani, N., & Chang, E. F. (2012). Selective cortical representation of attended speaker in multi-talker speech perception. Nature, 485, 233-236.
- Feldman, H., & Friston, K. J. (2010). Attention, uncertainty, and free-energy. Frontiers in Human Neuroscience, 4, 215.
