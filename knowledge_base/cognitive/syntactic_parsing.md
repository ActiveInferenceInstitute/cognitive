---
title: Syntactic Parsing
description: The mental process of assigning grammatical structure to a sentence as it is read or heard, including serial and parallel models, ambiguity resolution, reanalysis and probabilistic approaches.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - language
  - syntax
  - comprehension
semantic_relations:
  - type: relates
    links:
      - "[[sentence_processing]]"
      - "[[grammar]]"
      - "[[language_processing]]"
      - "[[language_comprehension]]"
      - "[[working_memory]]"
---

# Syntactic Parsing

## Overview
Syntactic parsing is the process of assigning a grammatical structure to a sentence, specifying which words group into phrases and what relations hold between them. Human parsing is incremental, so structure is built word by word, and it must cope with ambiguity, since many sentences allow more than one analysis at some point. The same term is used in computational linguistics for algorithms that return a syntactic tree for a text, a related but separate topic.

## Principles and models
- **Structure-first models.** The garden-path theory of Frazier and colleagues proposes that the parser initially builds one analysis using general principles. Minimal attachment prefers the simplest structure, and late closure prefers attaching new words to the current phrase. Frazier and Rayner (1982) used eye movements to show longer reading times when initial choices are wrong.
- **Constraint-based models.** Competing analyses are activated in parallel and weighted by lexical frequency, plausibility and context, so the same stimulus can show different initial preferences for different words.
- **Probabilistic and expectation-based models.** Analyses are ranked by probability under a grammar learned from experience, and difficulty relates to the probability of the incoming word or of the structures that must be abandoned.
- **Memory-based accounts.** Long-distance dependencies, such as those in relative clauses, are costly because earlier material must be held or retrieved, which links parsing to [[working_memory]].

## Reanalysis
When an initial analysis fails, the parser must revise it. The cost depends on how far the new structure departs from the old one. Electrophysiological studies associate syntactic anomalies and reanalysis with a late positive deflection called the P600, though its interpretation is debated.

## Relation to Active Inference
The link is a moderately developed interpretation. Parsing can be described as inferring the most probable hierarchical structure behind a word sequence, with predictions for upcoming categories and errors triggering revision. Probabilistic grammars used in computational parsing, such as those described by Jurafsky and Martin (2009), supply the statistical machinery. No account of parsing derived specifically from the free energy principle has been shown to outperform other probabilistic parsing models.

## Related concepts
- [[sentence_processing]] — the broader interpretation of sentences in real time
- [[grammar]] — the rule system the parser applies
- [[language_processing]] — the overall processing architecture
- [[language_comprehension]] — the setting in which parsing serves understanding
- [[working_memory]] — the resource constraining dependency integration

## References
- Frazier, L., & Fodor, J. D. (1978). The sausage machine: A new two-stage parsing model. Cognition, 6, 291-325.
- Frazier, L., & Rayner, K. (1982). Making and correcting errors during sentence comprehension: Eye movements in the analysis of structurally ambiguous sentences. Cognitive Psychology, 14, 178-210.
- Jurafsky, D., & Martin, J. H. (2009). Speech and Language Processing (2nd ed.). Pearson.
