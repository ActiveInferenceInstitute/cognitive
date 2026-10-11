---
title: Spaced Repetition
description: Reviewing material at increasing intervals to improve long-term retention, based on the spacing and testing effects, with scheduling methods, evidence and its modest relation to Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - memory
  - learning
  - education
semantic_relations:
  - type: relates
    links:
      - "[[memory_consolidation]]"
      - "[[retrieval]]"
      - "[[memory_formation]]"
      - "[[learning]]"
      - "[[encoding]]"
---

# Spaced Repetition

## Overview
Spaced repetition is a learning technique in which material is reviewed several times at intervals that typically grow longer, instead of being studied in a single block. It builds on two robust findings in memory research. The spacing effect is that distributed study produces better long-term retention than massed study of the same total duration. The testing effect is that actively retrieving information strengthens memory more than rereading it.

## Background
Hermann Ebbinghaus (1885) measured how rapidly learned nonsense syllables were forgotten and found that forgetting is steepest soon after learning and slows afterwards. He also found that repeated relearning reduced the effort needed to recall. Later reviews, such as Cepeda and colleagues (2006), synthesised many experiments and found that spacing benefits retention across a wide range of materials and ages. They also found that the best gap between study sessions depends on how long the material must be remembered: longer intended retention favours longer gaps.

## Scheduling
Software implementations assign each item a review date that depends on the learner's past success. A correct response lengthens the next interval, and a failure shortens it or resets it. Early algorithms used fixed multipliers per item, and newer ones fit a statistical model of each item's forgetting. Related practices are flashcards and the Leitner box system, in which cards move between boxes reviewed at different frequencies.

## Limits
Spacing improves retention but makes study feel harder, so learners often judge massed study as more effective than it is. The optimal schedule depends on the material, the learner and the retention goal, and many published benefits come from laboratory tasks with verbal items, so transfer to complex understanding is less certain.

## Relation to Active Inference
The link is weak. Spacing is a behavioural regularity that Active Inference does not explain directly, although consolidation is sometimes framed as refinement of a generative model. Choosing when to review an item can be described informally as deciding where uncertainty about one's own memory is highest, which parallels information-seeking, but this is an analogy and no model of spaced repetition has been established in the framework.

## Related concepts
- [[memory_consolidation]] — stabilisation processes thought to underlie the spacing effect
- [[retrieval]] — the act of recall whose practice drives the testing effect
- [[memory_formation]] — the stages by which memories are created
- [[learning]] — the broader process the technique supports
- [[encoding]] — the initial storage that reviews reinforce

## References
- Ebbinghaus, H. (1885). Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie. Duncker & Humblot.
- Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. Psychological Bulletin, 132, 354-380.
- Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. Psychological Science, 17, 249-255.
