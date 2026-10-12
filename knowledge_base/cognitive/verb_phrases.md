---
title: Verb Phrases
description: The syntactic constituent headed by a verb together with its complements and modifiers, its internal structure, verb subcategorisation, and its role in parsing and hierarchical models of language.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - syntax
  - grammar
  - psycholinguistics
semantic_relations:
  - type: relates
    links:
      - "[[grammar]]"
      - "[[noun_phrases]]"
      - "[[thematic_roles]]"
      - "[[sentence_processing]]"
      - "[[word_order]]"
---

# Verb Phrases

## Overview
A verb phrase is a syntactic constituent whose head is a verb. It contains the verb and the elements that complete or modify it, such as objects, complements and adverbials. In "The child [ate the apple quickly]", the bracketed string is a verb phrase headed by "ate", with "the apple" as a direct object and "quickly" as a modifier. In many frameworks the subject sits outside the verb phrase, and the verb phrase is combined with it to form a clause. Traditional grammars often include auxiliaries in the verb phrase, whereas generative grammar treats them as heads of separate projections.

## Structure
- **Head and complements.** A verb selects its complements according to its subcategorisation. Intransitive verbs ("sleep") take no object, transitive verbs ("devour") take one, ditransitive verbs ("give") take two, and others select a clause or a prepositional phrase ("believe that it rained", "rely on a friend"). The number and type of arguments is closely tied to [[thematic_roles]].
- **Phrase structure.** A rule such as VP → V (NP) (PP) captures the optionality and ordering of elements. In X-bar theory the verb phrase is a projection of V with intermediate levels that separate complements from adjuncts.
- **Evidence for the constituent.** Verb phrases can be coordinated ("sang and danced"), replaced by a pro-form ("John left, and Mary did too"), and elided. These tests show that the verb and its object group together more tightly than the subject and the verb.
- **Cross-linguistic variation.** The order of verb and object differs among languages, with head-initial verb phrases in English and head-final ones in Japanese or Turkish (see [[word_order]]).

## Processing and Active Inference
Comprehension requires attaching phrases to the verb as they arrive. Verb information is used early: in "The man hit the woman with the umbrella", the preference for attaching the prepositional phrase to the verb or to the noun depends on lexical and contextual cues. Studies of verb bias show that expectations about complement type, learned from experience, guide this choice. Such expectations resemble priors updated by evidence. Within Active Inference, language has been modelled with hierarchical generative models, and parsing can be described as inference over structure (Friston et al., 2020). The correspondence between verb phrases and levels of such a model is an analogy, since existing models are simple.

## Related concepts
- [[noun_phrases]] — the constituents that typically serve as verb arguments
- [[thematic_roles]] — the semantic roles that a verb assigns to its arguments
- [[grammar]] — the system defining phrases
- [[wh_movement]] — movement of a verb's argument to the front of a clause
- [[word_order]] — the arrangement of verb and arguments across languages
- [[sentence_processing]] — the incremental process of building phrases

## References
- Carnie, A. (2013). Syntax: A Generative Introduction (3rd ed.). Wiley-Blackwell.
- Huddleston, R., & Pullum, G. K. (2002). The Cambridge Grammar of the English Language. Cambridge University Press.
- Friston, K. J., Parr, T., Yufik, Y., Sajid, N., Price, C. J., & Holmes, E. (2020). Generative models, linguistic communication and active inference. Neuroscience & Biobehavioral Reviews, 118, 42-64.
