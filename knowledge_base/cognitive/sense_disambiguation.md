---
title: Sense Disambiguation
description: How comprehenders select the intended meaning of an ambiguous word in context, comparing exhaustive-access, selective-access and reordered-access models, with a note on its computational counterpart.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - lexical_ambiguity
  - semantics
  - comprehension
semantic_relations:
  - type: relates
    links:
      - "[[lexical_access]]"
      - "[[context_effects]]"
      - "[[semantic_memory]]"
      - "[[word_recognition]]"
      - "[[sentence_processing]]"
---

# Sense Disambiguation

## Overview
Many words have more than one meaning. "Bank" can name a financial institution or a river edge (homonymy, where unrelated meanings share a form), and "paper" can name a material, a document or a newspaper (polysemy, where senses are related). Sense disambiguation is the process by which a comprehender settles on the meaning intended, usually without noticing the ambiguity. The same term names a task in natural language processing, word sense disambiguation, surveyed by Navigli (2009).

## Models of access
Three families of accounts have dominated experimental work. Exhaustive access models hold that all meanings of a word are activated briefly, regardless of context, and context then selects among them. Swinney (1979) used cross-modal priming to show that both meanings of an ambiguous word primed related targets immediately after it was heard in a biasing sentence, with only the contextually appropriate one remaining after a short delay. Selective access models hold that a strong context can restrict activation to the appropriate meaning from the start. Reordered access models, supported by eye-tracking data, propose that all meanings are accessed, but that their activation depends on both their frequency and the context. In a study of fixation times, readers looked longer at an ambiguous word with two equally common meanings in neutral context, and also when context favoured the less frequent meaning of a biased word (Duffy et al., 1988). Rodd, Gaskell and Marslen-Wilson (2002) added that multiple unrelated meanings slow recognition through competition, whereas related senses do not slow it and may speed it.

## Open questions
Results depend on the task and on measurement timing. Cross-modal priming may overstate early activation, and strong constraining contexts are hard to equate across studies. Whether related senses are stored separately or share a core representation is also unresolved.

## Relation to Active Inference
The link is a moderate, conceptual one. Context acts as a prior over meanings, and selection resembles posterior inference, in keeping with predictive processing accounts of language. Established models of disambiguation, such as the reordered access model, were not developed within Active Inference and are compatible rather than derived from it.

## Related concepts
- [[lexical_access]] — retrieval of word information
- [[context_effects]] — contextual influence on perception and interpretation
- [[semantic_memory]] — the store of word meanings
- [[conceptual_priming]] — the technique used to measure activation
- [[lemma_selection]] — the corresponding choice in speech production

## References
- Swinney, D. A. (1979). Lexical access during sentence comprehension: (Re)consideration of context effects. Journal of Verbal Learning and Verbal Behavior, 18, 645-659.
- Duffy, S. A., Morris, R. K., & Rayner, K. (1988). Lexical ambiguity and fixation times in reading. Journal of Memory and Language, 27, 429-446.
- Rodd, J., Gaskell, G., & Marslen-Wilson, W. (2002). Making sense of semantic ambiguity: Semantic competition in lexical access. Journal of Memory and Language, 46, 245-266.
- Navigli, R. (2009). Word sense disambiguation: A survey. ACM Computing Surveys, 41(2), Article 10.
