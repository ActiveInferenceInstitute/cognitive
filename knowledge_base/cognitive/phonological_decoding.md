---
title: Phonological Decoding
description: The mapping of the incoming speech signal onto stored word forms, with the cohort, TRACE and Bayesian accounts and the debate over feedback, plus a note on the reading sense of the term.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - speech_perception
  - word_recognition
  - phonology
semantic_relations:
  - type: relates
    links:
      - "[[speech_perception]]"
      - "[[word_recognition]]"
      - "[[lexical_access]]"
      - "[[phonology]]"
      - "[[interactive_activation]]"
---

# Phonological Decoding

## Overview
In models of spoken language comprehension, phonological decoding is the process of converting the acoustic-phonetic information in speech into representations that can be matched to word forms stored in the mental lexicon. It is the step between prelexical analysis of the signal and access to meaning (see [[speech_perception]], [[lexical_access]]). In the reading literature the same term has a different sense: converting written letters into sounds, as in sounding out an unfamiliar word. This page covers the speech sense.

## Models
Speech arrives continuously and without reliable gaps between words, so decoding must be incremental and tolerate variability across speakers and contexts. In the cohort model, the first sounds of a word activate a set of candidates which is progressively narrowed as further input arrives, until a unique word remains (Marslen-Wilson & Welsh, 1978). TRACE, an interactive activation network, has layers for features, phonemes and words, with excitation passing in both directions and competition within layers (McClelland & Elman, 1986). Shortlist and its Bayesian successor treat recognition as selecting the most probable word sequence given the input, with competition between overlapping candidates but without feedback from words to phonemes (Norris & McQueen, 2008).

## Debate over feedback
The main disagreement is whether lexical knowledge feeds back to alter prelexical processing. Interactive models explain lexical effects on phoneme identification as top-down influence. Norris, McQueen and Cutler (2000) argued that feedback is never necessary, since the same effects can arise from merging lexical and phonemic information at a decision stage, and that feedback may not improve recognition. Subsequent work on perceptual learning, in which listeners retune phoneme categories using lexical context, is cited on both sides. The matter is not resolved.

## Relation to Active Inference
The connection is substantial in spirit. Recognition as Bayesian inference over words given acoustic evidence matches the generative framing, and the feedback debate parallels the question of how top-down predictions and bottom-up prediction errors interact in predictive coding. Bayesian models of this kind predate and are independent of Active Inference, so they are compatible rather than derived from it.

## Related concepts
- [[speech_perception]] — the earlier stage of analysis
- [[word_recognition]] — the outcome of decoding
- [[phonology]] — the sound system that supplies the units
- [[interactive_activation]] — the framework behind TRACE
- [[onset_rhyme]] — sub-syllable units used in the analysis of word forms

## References
- Marslen-Wilson, W. D., & Welsh, A. (1978). Processing interactions and lexical access during word recognition in continuous speech. Cognitive Psychology, 10, 29-63.
- McClelland, J. L., & Elman, J. L. (1986). The TRACE model of speech perception. Cognitive Psychology, 18, 1-86.
- Norris, D., McQueen, J. M., & Cutler, A. (2000). Merging information in speech recognition: Feedback is never necessary. Behavioral and Brain Sciences, 23, 299-325.
- Norris, D., & McQueen, J. M. (2008). Shortlist B: A Bayesian model of continuous speech recognition. Psychological Review, 115, 357-395.
