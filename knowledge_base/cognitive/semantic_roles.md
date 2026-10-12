---
title: Semantic Roles
description: The roles that participants play in the event described by a clause, such as agent, patient and instrument, with competing inventories, psycholinguistic evidence and their use in computational labelling.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - semantics
  - sentence_comprehension
  - linguistics
semantic_relations:
  - type: relates
    links:
      - "[[thematic_roles]]"
      - "[[predicate_argument]]"
      - "[[verb_phrases]]"
      - "[[sentence_processing]]"
      - "[[language_comprehension]]"
---

# Semantic Roles

## Overview
Semantic roles are the relations between a predicate and the participants in the event or state it describes. In "Ana opened the door with a key", Ana is the agent, who initiates the action, the door is the patient or theme, which undergoes it, and the key is the instrument. The roles capture who did what to whom independently of the syntactic position, so that "The door was opened by Ana" keeps the same roles with a different grammatical subject. The label thematic roles is widely used for the same notion, with [[thematic_roles]] and [[predicate_argument]] treating neighbouring aspects.

## Inventories
There is no agreed list. Fillmore (1968) introduced case roles, including agent, instrument, object and location. Later frameworks differ in number and granularity, from a few broad roles to very many verb-specific ones. Dowty (1991) proposed replacing discrete roles with two clusters of properties, proto-agent properties such as volition and causation and proto-patient properties such as change of state, so that roles become matters of degree. The Proposition Bank annotates numbered arguments for each verb sense instead of a universal list of labels (Palmer et al., 2005). Which approach is correct partly depends on the purpose, whether formal semantics, linking of roles to syntax, or processing.

## Evidence from comprehension
Roles carry expectations. McRae, Ferretti and Amyote (1997) argued that knowledge of roles is part of the concepts associated with verbs, and found that a noun's fit with a verb's agent or patient role affects how quickly it is processed and how ambiguities are resolved. This supports the view that role information is used early in comprehension. How far such effects reflect discrete role categories, as opposed to general event knowledge, is disputed.

## Computational use
Semantic role labelling is a task in natural language processing: determining, for each predicate in a sentence, which spans fill which roles. Gildea and Jurafsky (2002) described an early statistical system for this task using FrameNet roles. The task is usually evaluated on annotated corpora and is distinct from claims about human cognition.

## Relation to Active Inference
No established link. Roles may be seen as structure in a generative model of events, but this is a loose analogy.

## Related concepts
- [[thematic_roles]] — the same notion under its other common name
- [[predicate_argument]] — the structure in which roles are assigned
- [[verb_phrases]] — the syntactic units headed by predicates
- [[sentence_processing]] — where role assignment occurs incrementally
- [[lexical_access]] — retrieval of verb information that supplies expected roles

## References
- Fillmore, C. J. (1968). The case for case. In E. Bach & R. T. Harms (Eds.), Universals in Linguistic Theory (pp. 1-88). Holt, Rinehart and Winston.
- Dowty, D. (1991). Thematic proto-roles and argument selection. Language, 67, 547-619.
- McRae, K., Ferretti, T. R., & Amyote, L. (1997). Thematic roles as verb-specific concepts. Language and Cognitive Processes, 12, 137-176.
- Gildea, D., & Jurafsky, D. (2002). Automatic labeling of semantic roles. Computational Linguistics, 28, 245-288.
- Palmer, M., Gildea, D., & Kingsbury, P. (2005). The Proposition Bank: An annotated corpus of semantic roles. Computational Linguistics, 31, 71-106.
