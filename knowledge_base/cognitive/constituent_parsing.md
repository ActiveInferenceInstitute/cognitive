---
title: Constituent Parsing
description: Analysis of a sentence into nested phrases such as noun and verb phrases, in linguistic theory and in human sentence comprehension, including garden-path effects and probabilistic models.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - syntax
  - language
  - parsing
semantic_relations:
  - type: relates
    links:
      - "[[syntactic_parsing]]"
      - "[[sentence_processing]]"
      - "[[grammar]]"
      - "[[noun_phrases]]"
      - "[[verb_phrases]]"
---

# Constituent Parsing

## Overview
Constituent parsing, also called phrase-structure parsing, assigns a sentence a hierarchical structure of constituents: groups of words that behave as units. In "the dog chased a cat", "the dog" and "chased a cat" are constituents, the latter containing "a cat" as a further constituent. Such analyses are usually drawn as labelled trees whose nodes are categories such as sentence, noun phrase and verb phrase.

## Linguistic basis
Constituency is motivated by tests. A string is likely a constituent if it can be replaced by a pro-form, moved as a unit, appear alone as an answer to a question, or be coordinated with a similar string. Phrase-structure grammars capture this with rewrite rules, such as a sentence consisting of a noun phrase followed by a verb phrase. Dependency grammars offer an alternative that represents head-dependent relations between words and not nested phrases.

## Parsing in humans
Comprehenders build structure incrementally, often before the end of the sentence. Temporarily ambiguous sentences expose this. In "the horse raced past the barn fell", readers commit to a main-verb reading of "raced" and must reanalyse. Early accounts such as Frazier's garden-path model proposed that the parser first adopts the simplest structure, using principles such as minimal attachment, whereas constraint-based accounts propose that many sources of information, including word frequency, plausibility and context, compete from the start. Eye-movement studies show longer fixations and regressions at disambiguating words.

## Computational parsing
In computational linguistics, probabilistic context-free grammars score alternative trees, and chart parsers or neural models search for the best one. Expectation-based theories of comprehension link processing difficulty to surprisal, the negative log probability of a word given its preceding context.

## Relation to Active Inference
Incremental parsing is naturally described as predictive inference: a hierarchical model generates expectations about upcoming structure, and difficulty at a garden-path sentence resembles a large prediction error followed by belief revision. Surprisal is a close relative of that idea. Active Inference treatments of sentence structure exist, but they are an emerging line of work and not yet a standard account of parsing.

## Related concepts
- [[syntactic_parsing]] — the broader class of structure-assigning procedures
- [[sentence_processing]] — the psycholinguistic study of comprehension in real time
- [[grammar]] — the rules that define permitted structures
- [[noun_phrases]] — one of the main constituent types
- [[verb_phrases]] — the predicate constituent
- [[wh_movement]] — a case where constituents appear away from their canonical position

## References
- Chomsky, N. (1957). Syntactic Structures. Mouton.
- Frazier, L., & Rayner, K. (1982). Making and correcting errors during sentence comprehension: Eye movements in the analysis of structurally ambiguous sentences. Cognitive Psychology, 14, 178-210.
- Levy, R. (2008). Expectation-based syntactic comprehension. Cognition, 106, 1126-1177.
