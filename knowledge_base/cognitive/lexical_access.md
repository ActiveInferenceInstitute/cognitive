---
title: Lexical Access
description: The process of retrieving word forms and meanings from the mental lexicon during listening, reading and speaking, including frequency effects, competition between candidates, and Bayesian accounts.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - language
  - word_recognition
  - memory_retrieval
semantic_relations:
  - type: relates
    links:
      - "[[word_recognition]]"
      - "[[semantic_memory]]"
      - "[[language_processing]]"
      - "[[speech_perception]]"
      - "[[aphasia]]"
aliases:
  - lexical_retrieval
---

# Lexical Access

## Overview
Lexical access is the retrieval of information about words from long-term memory. In comprehension, a sound or letter string is mapped onto a stored entry whose form and meaning become available. In production, a concept is mapped onto a word and then its sound pattern. The store of such entries is called the mental lexicon. Access is rapid, typically completed within a few hundred milliseconds, and is mostly automatic.

## Mechanisms
Several robust findings constrain theories.
- **Frequency**: common words are recognised and produced faster than rare ones.
- **Priming**: a word is processed faster after a related word, a pattern usually explained by spreading activation among lexical and semantic representations (Collins & Loftus, 1975).
- **Competition**: during spoken word recognition, input activates many candidates consistent with the sounds heard so far, and these compete until one is selected. The cohort model (Marslen-Wilson, 1987) and the connectionist TRACE model (McClelland & Elman, 1986) formalise this incremental process.
- **Production stages**: influential models separate selection of an abstract lexical item from retrieval of its phonological form (Levelt, 1989; Levelt, Roelofs, & Meyer, 1999). Tip-of-the-tongue states, in which meaning is available but form is not, support such a division.

Disorders show the same architecture at work: word-finding difficulty is prominent in [[aphasia]], and children with [[specific_language_impairment]] often learn words slowly.

## Relation to Active Inference
The connection is genuine but mostly through Bayesian modelling of language rather than through Active Inference proper. Probabilistic models treat lexical access as inference over candidate words using priors, such as word frequency and sentence context, combined with noisy sensory evidence; the Bayesian Reader and Shortlist B (Norris, 2006; Norris & McQueen, 2008) are examples. Predictive-processing accounts extend this by proposing that context-driven expectations pre-activate likely words, so that confirmed predictions need little further processing and violated ones generate prediction errors, as in the N400 literature. Whether such effects reflect prediction or easier integration is still debated. Active Inference adds that the choice of word in production can itself be cast as policy selection, but this is an emerging formulation rather than a settled one. See [[predictive_processing]].

## Related concepts
- [[word_recognition]] — the comprehension side of lexical access
- [[semantic_memory]] — the knowledge that retrieved entries connect to
- [[speech_perception]] — the acoustic input stage preceding access
- [[language_processing]] — the larger system in which access operates
- [[predictive_processing]] — context-based expectation of upcoming words

## References
- Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. Psychological Review, 82, 407-428.
- Marslen-Wilson, W. D. (1987). Functional parallelism in spoken word-recognition. Cognition, 25, 71-102.
- Levelt, W. J. M. (1989). Speaking: From Intention to Articulation. MIT Press.
- Norris, D. (2006). The Bayesian reader: Explaining word recognition as an optimal Bayesian decision process. Psychological Review, 113, 327-357.
