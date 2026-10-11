---
title: Sentence Processing
description: How listeners and readers incrementally interpret sentence structure and meaning, including parsing strategies, garden-path effects, constraint-based accounts and surprisal, with links to prediction.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - language
  - comprehension
  - prediction
semantic_relations:
  - type: relates
    links:
      - "[[syntactic_parsing]]"
      - "[[language_comprehension]]"
      - "[[psycholinguistics]]"
      - "[[working_memory]]"
      - "[[predictive_processing]]"
---

# Sentence Processing

## Overview
Sentence processing is the set of mental operations by which a listener or reader assigns structure and meaning to a sentence as it unfolds. Because language arrives word by word, interpretation is largely incremental: comprehenders commit to partial analyses before the sentence ends. The central questions are which information guides these decisions and how errors are detected and repaired.

## Ambiguity and garden paths
Temporary ambiguity reveals the process. In "The horse raced past the barn fell", most readers initially analyse "raced" as a main verb and are forced to reanalyse when "fell" appears. Eye-tracking studies show longer reading times and regressions at such points of disambiguation. Two broad families of theory account for these effects.
- **Serial, structure-first accounts.** Frazier and Fodor's (1978) model proposes that the parser builds a single analysis guided by general principles such as minimal attachment, and revises it if later input contradicts it.
- **Constraint-based accounts.** MacDonald, Pearlmutter and Seidenberg (1994) argued that multiple analyses are weighed in parallel using lexical, contextual and frequency information, with the competing analyses activated in proportion to their support.

## Prediction and surprisal
A growing body of work shows that comprehenders anticipate upcoming material. Expectation-based theories propose that processing difficulty at a word reflects how unexpected it is given the preceding context, formalised as surprisal, the negative log probability of the word given its context. Levy (2008) developed this account for syntactic comprehension, and surprisal estimated from language models predicts reading times across many studies. Memory-based accounts add that integrating distant dependencies, as in nested clauses, taxes working memory.

## Relation to Active Inference
The link is an active research area. Incremental prediction and revision fit predictive processing, in which comprehension is inference over a hierarchy of linguistic structures with prediction errors guiding revision. Surprisal can be related to variational free energy as a measure of unexpected input. However, surprisal-based accounts are also compatible with non-hierarchical probabilistic models, so reading-time data alone do not isolate the Active Inference proposal.

## Related concepts
- [[syntactic_parsing]] — the structure-building component
- [[language_comprehension]] — the wider process of understanding language
- [[psycholinguistics]] — the field that studies these processes
- [[working_memory]] — the memory resource used to integrate dependencies
- [[predictive_processing]] — the framework linking surprise and prediction

## References
- Frazier, L., & Fodor, J. D. (1978). The sausage machine: A new two-stage parsing model. Cognition, 6, 291-325.
- MacDonald, M. C., Pearlmutter, N. J., & Seidenberg, M. S. (1994). The lexical nature of syntactic ambiguity resolution. Psychological Review, 101, 676-703.
- Levy, R. (2008). Expectation-based syntactic comprehension. Cognition, 106, 1126-1177.
