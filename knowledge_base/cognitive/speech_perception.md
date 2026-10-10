---
title: Speech Perception
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - language
  - audition
  - perception
aliases:
  - speech_recognition
  - phoneme_recognition
semantic_relations:
  - type: relates
    links:
      - "[[auditory_processing]]"
      - "[[language_processing]]"
      - "[[perceptual_inference]]"
---

# Speech Perception

## Overview
Speech perception is the process of mapping a continuously varying acoustic signal onto linguistic units such as phonemes, syllables and words. Listeners do this rapidly and with remarkable tolerance to speaker, accent, rate and noise, even though the signal contains no clean boundaries between sounds.

## Mechanism
Two properties define the problem. First, the lack of invariance: there is no simple acoustic feature that always signals a given phoneme, because coarticulation makes each sound depend on its neighbors. Second, segmentation: words in running speech are not separated by reliable silences.

Perception of some contrasts is categorical, with sharp identification boundaries along a continuum such as voice onset time (the delay between a stop's release and the start of voicing) and better discrimination across than within categories. Listeners exploit multiple cues, lexical knowledge and context; for example, the same ambiguous sound is heard differently depending on the word it appears in.

Visual information also contributes. In the McGurk effect, a video of one syllable combined with the audio of another yields a percept that matches neither (McGurk and MacDonald). The motor theory proposed by Liberman and colleagues held that listeners perceive intended articulatory gestures, an idea that remains debated.

Neurally, Hickok and Poeppel's dual-stream model places sound-to-meaning mapping in a ventral stream through the temporal lobe and sensorimotor mapping in a dorsal stream linking auditory regions to frontal articulatory systems. Models of word recognition such as TRACE use interactive activation between acoustic-feature, phoneme and word levels, with top-down influence from word to phoneme.

## Role in Active Inference
Speech perception is a strong case for hierarchical inference: higher levels (words, syntax, topic) supply predictions about lower-level sounds, and ambiguous input is resolved toward expected interpretations. Lexical effects on phoneme identification and the benefit of context in noise are consistent with this. Active-inference models of communication treat listening and speaking as complementary processes in one generative model, and the precision of predictions can be tuned to the reliability of the signal.

Interactive-activation and Bayesian word-recognition models make similar predictions without invoking free-energy minimization, and whether top-down influence changes perception itself or only decisions has been disputed, so the support is partial.

## Related concepts
- [[auditory_processing]] — acoustic and cortical analysis preceding speech-specific stages
- [[language_processing]] — higher linguistic processing
- [[cross_modal_perception]] — audiovisual effects such as the McGurk effect
- [[multisensory_integration]] — combining lip movements and sound
- [[perceptual_inference]] — inference framing of ambiguous input
- [[neuroprosthetics]] — cochlear implants and speech outcomes

## References
- Liberman, A. M., Cooper, F. S., Shankweiler, D. P., & Studdert-Kennedy, M. (1967). Perception of the speech code. Psychological Review.
- McGurk, H., & MacDonald, J. (1976). Hearing lips and seeing voices. Nature.
- McClelland, J. L., & Elman, J. L. (1986). The TRACE model of speech perception. Cognitive Psychology.
- Hickok, G., & Poeppel, D. (2007). The cortical organization of speech processing. Nature Reviews Neuroscience.
