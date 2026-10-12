---
title: Parsing
description: How comprehenders assign grammatical structure to incoming sentences, garden-path evidence, serial versus constraint-based accounts, and expectation-based models of processing difficulty.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - psycholinguistics
  - sentence_comprehension
  - syntax
semantic_relations:
  - type: relates
    links:
      - "[[syntactic_parsing]]"
      - "[[sentence_processing]]"
      - "[[grammar]]"
      - "[[language_comprehension]]"
      - "[[predictive_processing]]"
---

# Parsing

## Overview
In psycholinguistics, parsing is the process by which a reader or listener assigns syntactic structure to a sentence, incrementally and almost always before the sentence ends. Structure is needed to decide who did what to whom (see [[predicate_argument]]). The same word is also used in computer science for algorithms that analyse strings against a formal grammar; this page concerns the human process, and [[syntactic_parsing]] covers the closely related topic from the language-processing side.

## Garden paths and the two main accounts
Temporary ambiguity is the main testing ground. In "The horse raced past the barn fell", readers initially take "raced" as a main verb and are forced to reanalyse, a garden-path effect. Eye-movement studies show longer fixations and regressions at the point of disambiguation (Frazier & Rayner, 1982). Two families of accounts explain such effects. In two-stage models, a first stage builds a single structure using only syntactic information, guided by principles such as preferring the simplest attachment, and later stages revise it using meaning and context. In constraint-based models, multiple analyses are weighed in parallel, using lexical statistics, plausibility and context from the start; MacDonald, Pearlmutter and Seidenberg (1994) argued that much apparent syntactic bias comes from lexical information about verbs. Experiments have found context effects early enough to trouble strictly syntax-first accounts, but proponents of the first family dispute the interpretation, so the issue is not closed.

## Expectation-based models
A more recent line holds that processing difficulty reflects how unexpected a word is given the preceding structure. Surprisal theory proposes that the cost of a word is proportional to its negative log probability in context, and Levy (2008) showed that such a quantity accounts for a range of reading-time effects.

## Relation to Active Inference
The connection is moderate and conceptual. Incremental parsing treated as inference over possible structures, updated as each word arrives, resembles predictive processing, and surprisal can be read as a prediction error measure. Whether surprisal is the cause of difficulty or a correlate of other costs is still debated, and no specific Active Inference parser is established.

## Related concepts
- [[syntactic_parsing]] — the same topic in the language-processing literature
- [[sentence_processing]] — the wider comprehension process
- [[grammar]] — the structures being assigned
- [[serial_processing]] — the staged assumption in two-stage models
- [[predictive_processing]] — the general framework that expectation-based accounts resemble

## References
- Frazier, L., & Rayner, K. (1982). Making and correcting errors during sentence comprehension: Eye movements in the analysis of structurally ambiguous sentences. Cognitive Psychology, 14, 178-210.
- MacDonald, M. C., Pearlmutter, N. J., & Seidenberg, M. S. (1994). The lexical nature of syntactic ambiguity resolution. Psychological Review, 101, 676-703.
- Levy, R. (2008). Expectation-based syntactic comprehension. Cognition, 106, 1126-1177.
