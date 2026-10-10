---
title: Word Recognition
description: "Identifying written or spoken words and accessing their stored meanings, covering frequency and context effects, the word superiority effect, interactive-activation and cohort models, and Bayesian and predictive accounts."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - language
  - reading
  - lexical-access
semantic_relations:
  - type: relates
    links:
      - "[[language_processing]]"
      - "[[speech_perception]]"
      - "[[semantic_memory]]"
      - "[[perceptual_inference]]"
---

# Word Recognition

## Overview
Word recognition is the process by which a perceiver maps a visual or auditory signal onto a stored lexical entry and its meaning, usually in a few hundred milliseconds. It involves resolving ambiguity from noise, from competing word candidates, and from variation in how the word is produced. Models differ in whether information flows only forward or also back from higher levels.

## Findings and models
- **Frequency and neighbourhood.** Common words are recognised faster. Words with many similar neighbours are affected by competition, and the direction of the effect differs across tasks.
- **Context and priming.** A preceding related word or sentence context speeds recognition (semantic priming), and predictable words are read with shorter fixations.
- **Word superiority.** A letter is identified more accurately inside a word than alone or in a nonword (Reicher), which suggests that word-level knowledge feeds back to letter perception.
- **Interactive activation.** McClelland and Rumelhart modelled this as a network with letter and word levels connected by excitatory and inhibitory links, in which activation spreads both ways. TRACE applies the architecture to spoken words ([[speech_perception]]).
- **Cohort model.** In speech, the incoming sound activates all words consistent with the onset, and the set narrows as more input arrives until a unique candidate remains (Marslen-Wilson).
- **Neural basis.** A left ventral occipitotemporal region (the visual word form area) responds to written words; superior temporal and inferior frontal regions support spoken words and access to meaning ([[superior_temporal]], [[language_processing]]).

## Role in Active Inference
Word recognition is a clear case of perceptual inference under strong priors. Norris's Bayesian Reader shows that frequency and context effects follow from optimal inference over a lexicon, and predictive coding accounts of reading, such as the interactive account of ventral occipitotemporal cortex, propose that top-down lexical and semantic predictions are compared with visual input, with prediction errors driving the lower-level response ([[top_down]], [[prediction_error]]). These models share the idea of hierarchical inference but are not specific to the free-energy principle, and the neural claims are under debate.

## Related concepts
- [[language_processing]] - the broader setting
- [[speech_perception]] - spoken input to words
- [[semantic_memory]] - the stored meanings accessed
- [[phonological_loop]] - holds phonological forms during processing
- [[pattern_recognition]] - the general computational problem

## References
- Reicher, G. M. (1969). Perceptual recognition as a function of meaningfulness of stimulus material. Journal of Experimental Psychology.
- McClelland, J. L., & Rumelhart, D. E. (1981). An interactive activation model of context effects in letter perception: Part 1. Psychological Review.
- Norris, D. (2006). The Bayesian reader: explaining word recognition as an optimal Bayesian decision process. Psychological Review.
- Price, C. J., & Devlin, J. T. (2011). The interactive account of ventral occipitotemporal contributions to reading. Trends in Cognitive Sciences.
