---
title: Categorical Perception
description: The tendency to perceive continuously varying stimuli as belonging to discrete categories, with sharper discrimination across category boundaries than within them, and Bayesian accounts of the effect.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - perception
  - speech
  - categorization
semantic_relations:
  - type: relates
    links:
      - "[[speech_perception]]"
      - "[[category_learning]]"
      - "[[perceptual_inference]]"
      - "[[phonology]]"
      - "[[psychophysics]]"
---

# Categorical Perception

## Overview
Categorical perception occurs when a physical continuum is experienced as a set of distinct categories. Two stimuli that differ by a fixed physical amount are discriminated easily if they fall on opposite sides of a category boundary, and poorly if they fall within one category. The effect was first documented in speech and has since been reported for colour, facial expression, and learned visual categories.

## Speech and the classic findings
Liberman, Harris, Hoffman and Griffith (1957) synthesised a continuum of stop consonants such as /b/, /d/ and /g/ and found that listeners labelled the steps in abrupt transitions, and discriminated pairs best when the pair straddled a labelling boundary. A well-studied case is voice onset time, which separates voiced from voiceless stops such as /b/ and /p/. Strong forms of the claim, in which within-category differences are not perceived at all, are not supported: listeners show graded sensitivity within categories, for example in reaction times and goodness ratings. The effect also appears in some non-human animals, so it is not unique to language.

## Explanations
Accounts differ on how much is innate and how much learned. Learned categorical perception, in which training on a category boundary increases between-category discrimination, shows that experience can produce the pattern (Harnad, 1987). Bayesian accounts treat the effect as optimal inference when stimuli are drawn from category distributions: noisy percepts are shrunk toward the mean of the most probable category, which warps perceptual space near category centres. Feldman, Griffiths and Morgan (2009) used this idea to explain the perceptual magnet effect in vowels.

## Relation to Active Inference
The Bayesian account is the same inferential structure that predictive processing assumes for perception generally: a higher-level categorical hypothesis acts as a prior on lower-level sensory estimates, so that interpretation is pulled toward the category. This is a close formal analogy and has direct support in the Bayesian modelling of speech categories, but it is not a specific Active Inference result. Learning the categories themselves corresponds to updating a hierarchical generative model.

## Related concepts
- [[speech_perception]] — the domain where the effect was discovered
- [[category_learning]] — how boundaries are acquired
- [[perceptual_inference]] — perception as inference over causes
- [[phonology]] — phoneme categories as linguistic units
- [[language_processing_expanded]] — the wider language-processing context
- [[context_effects]] — contextual shifts of category boundaries

## References
- Liberman, A. M., Harris, K. S., Hoffman, H. S., & Griffith, B. C. (1957). The discrimination of speech sounds within and across phoneme boundaries. Journal of Experimental Psychology, 54, 358-368.
- Harnad, S. (Ed.). (1987). Categorical Perception: The Groundwork of Cognition. Cambridge University Press.
- Feldman, N. H., Griffiths, T. L., & Morgan, J. L. (2009). The influence of categories on perception: Explaining the perceptual magnet effect as optimal statistical inference. Psychological Review, 116, 752-782.
