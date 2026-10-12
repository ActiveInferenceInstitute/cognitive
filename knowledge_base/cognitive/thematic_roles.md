---
title: Thematic Roles
description: The semantic roles that verbs assign to their arguments, such as agent and patient, major theoretical treatments, and their use in sentence comprehension and event models.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - semantics
  - syntax
  - sentence_comprehension
semantic_relations:
  - type: relates
    links:
      - "[[sentence_processing]]"
      - "[[compositional_semantics]]"
      - "[[verb_phrases]]"
      - "[[grammar]]"
      - "[[word_order]]"
---

# Thematic Roles

## Overview
Thematic roles, also called semantic roles or theta roles, describe the part that each participant plays in the event a verb denotes: who did what to whom, with what, for whom and where. In "Mary opened the door with a key", Mary is the agent, the door is the patient or theme, and the key is the instrument. The roles link meaning to syntax, since a verb determines which arguments it takes and which role each receives.

## Inventories and theories
Commonly proposed roles include agent, patient, theme, experiencer, instrument, source, goal, location and beneficiary. Their number and definitions are disputed.
- **Case grammar.** Fillmore (1968) proposed a small set of deep cases, such as agentive, instrumental and objective, that determine surface grammatical relations.
- **Theta theory.** In Government and Binding theory each argument must receive exactly one theta role, and each role assigned by a predicate must go to exactly one argument (the theta criterion).
- **Proto-roles.** Dowty (1991) argued that discrete roles are hard to define and proposed two clusters of entailments. Proto-agent properties include volitional involvement, sentience, causing an event or change, and movement. Proto-patient properties include undergoing a change of state, being causally affected, and being stationary relative to another participant. An argument is mapped to subject or object according to which cluster it has more of.
- **Frame semantics.** Roles are specific to situations, such as the buyer and seller of a commercial transaction, and are grouped into frames.

Roles map onto syntax in regular but not exclusive ways. Agents are typically subjects in active clauses, while passives and alternations such as "The key opened the door" reassign roles to different positions.

## Comprehension and Active Inference
Listeners appear to assign roles incrementally. Preferences such as treating an initial noun phrase as an agent can lead to misanalysis in sentences like "The horse raced past the barn fell", and verb meaning constrains which roles are expected next. Understanding a sentence can be described as inferring the event that generated it, with the roles as the slots of a hidden event representation, and Friston and colleagues (2020) have formulated linguistic communication in generative-model terms. The use of thematic roles in that setting is a description at the level of analogy; no model has been shown to derive the role inventory from Active Inference.

## Related concepts
- [[sentence_processing]] — the real-time process that assigns roles
- [[verb_phrases]] — the phrases in which a verb licenses its arguments
- [[compositional_semantics]] — how roles combine into sentence meaning
- [[word_order]] — a cue to role assignment in many languages
- [[grammar]] — the system relating roles to structure
- [[noun_phrases]] — the usual bearers of thematic roles

## References
- Fillmore, C. J. (1968). The case for case. In E. Bach & R. T. Harms (Eds.), Universals in Linguistic Theory (pp. 1-88). Holt, Rinehart and Winston.
- Dowty, D. (1991). Thematic proto-roles and argument selection. Language, 67, 547-619.
- Friston, K. J., Parr, T., Yufik, Y., Sajid, N., Price, C. J., & Holmes, E. (2020). Generative models, linguistic communication and active inference. Neuroscience & Biobehavioral Reviews, 118, 42-64.
