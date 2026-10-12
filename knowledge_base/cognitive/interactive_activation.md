---
title: Interactive Activation
description: A connectionist framework in which processing levels exchange excitation and inhibition in both directions, its model of word perception, and its relation to predictive coding.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - connectionism
  - word_recognition
  - language
semantic_relations:
  - type: relates
    links:
      - "[[word_recognition]]"
      - "[[lexical_access]]"
      - "[[speech_perception]]"
      - "[[hierarchical_processing]]"
      - "[[predictive_coding]]"
---

# Interactive Activation

## Overview
Interactive activation is a connectionist framework in which processing units are arranged in levels, and activation flows between levels in both directions. Units that are mutually consistent excite one another, and competing units within a level inhibit one another. The framework was introduced by McClelland and Rumelhart (1981) to account for how context influences the perception of letters, and it became an influential alternative to purely bottom-up or purely serial accounts of recognition.

## The word perception model
The original model has three levels: visual feature units, letter units for each position in a four-letter string, and word units. Feature units excite the letters that contain them and inhibit incompatible letters; letters excite the words containing them in the right positions and inhibit the others; word units inhibit each other, and send excitatory feedback to the letters they contain. The model explains the word superiority effect, in which a letter is identified more accurately within a word than alone or in a nonword (Rumelhart and McClelland, 1982). Active word candidates feed activation back to their letters and strengthen them, so context helps perception without a separate decision stage. Activation builds up over time in a cascaded, parallel way.

## Extensions and debate
McClelland and Elman (1986) applied the same principles to spoken word recognition in the TRACE model, with feature, phoneme and word levels, and used it to simulate lexical effects on phoneme identification (see [[speech_perception]]). A longstanding debate concerns whether feedback from words to lower levels is really needed, or whether lexical knowledge only biases decisions after perceptual processing. Models without feedback can account for many of the data, and the empirical case for lexical effects on perception itself remains discussed. Interactive models are also criticised for lacking a principled account of learning and probability.

## Relation to Active Inference
Interactive activation is a close precursor in spirit to predictive-processing accounts, which also assume bidirectional message passing and hierarchical competition between hypotheses. The two differ in what is passed. In predictive coding, descending messages carry predictions and ascending messages carry prediction errors, whereas in interactive activation both directions carry activation. Later work argued that interactive activation can approximate probabilistic inference under suitable assumptions. The relation is therefore one of resemblance, and the models are not equivalent.

## Related concepts
- [[word_recognition]] — the domain where the model was first applied
- [[lexical_access]] — retrieval of word representations
- [[speech_perception]] — the TRACE extension
- [[hierarchical_processing]] — levels of representation
- [[predictive_coding]] — a related bidirectional scheme

## References
- McClelland, J. L., & Rumelhart, D. E. (1981). An interactive activation model of context effects in letter perception: Part 1. An account of basic findings. Psychological Review, 88, 375-407.
- Rumelhart, D. E., & McClelland, J. L. (1982). An interactive activation model of context effects in letter perception: Part 2. The contextual enhancement effect and some tests and extensions of the model. Psychological Review, 89, 60-94.
- McClelland, J. L., & Elman, J. L. (1986). The TRACE model of speech perception. Cognitive Psychology, 18, 1-86.
