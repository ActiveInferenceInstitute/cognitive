---
title: Serial Processing
description: Processing in which operations or stages are carried out one after another, how serial and parallel architectures are distinguished experimentally, and where serial assumptions persist in language and attention theory.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - cognitive_architecture
  - processing_models
  - reaction_time
semantic_relations:
  - type: relates
    links:
      - "[[parallel_processing]]"
      - "[[serial_bottleneck]]"
      - "[[interactive_activation]]"
      - "[[visual_search]]"
      - "[[language_processing]]"
---

# Serial Processing

## Overview
In cognitive psychology, serial processing means that processing operations are performed one at a time, with each one completing before the next begins, or that a series of stages operate in strict order. It contrasts with [[parallel_processing]], in which several operations proceed at once. The distinction applies at different scales: whether items in a display are examined one by one, whether stages of a task such as perception, decision and response occur in sequence, and whether levels of language processing pass completed results forward.

## Stage logic
The idea has roots in Donders' subtraction method of the nineteenth century, which assumed that the time for a complex task is the sum of the times of its component stages. Sternberg (1969) introduced the additive factors method: if two experimental factors have additive effects on reaction time, they probably influence different stages, whereas interacting factors affect a common stage. This method assumes that stages operate serially and that each hands over a finished output, an assumption which cascade models reject.

## Distinguishing serial from parallel
Serial and parallel models can often mimic each other's predictions. Townsend (1990) showed that some standard tests cannot tell them apart without additional assumptions, because a parallel system with limited capacity can produce the same reaction-time patterns as a serial one. In visual search, the feature integration theory proposed serial scanning for conjunction targets (Treisman & Gelade, 1980), but later work, including accounts in which search is guided by features and capacity-limited parallel models, questioned a strict serial/parallel dichotomy. In dual-task performance, response selection appears to form a central bottleneck that handles one task at a time, although the bottleneck's cause and universality are debated (Pashler, 1994; see [[serial_bottleneck]]). In language, modular theories of parsing and production assume ordered stages, whereas interactive accounts such as [[interactive_activation]] allow overlap and feedback.

## Relation to Active Inference
Message passing in hierarchical generative models is generally treated as simultaneous across levels, with ascending prediction errors and descending predictions exchanged in parallel. Serial limits would then reflect constraints of the model, such as the need to commit to one policy at a time, not a fundamental architecture. This is an interpretation, not a derived result.

## Related concepts
- [[parallel_processing]] — the contrasting architecture
- [[serial_bottleneck]] — a limitation on simultaneous task performance
- [[visual_search]] — a paradigm for testing serial scanning
- [[dual_tasks]] — the paradigm used to reveal bottlenecks
- [[language_processing]] — where staged models are used

## References
- Sternberg, S. (1969). The discovery of processing stages: Extensions of Donders' method. Acta Psychologica, 30, 276-315.
- Townsend, J. T. (1990). Serial vs. parallel processing: Sometimes they look like Tweedledum and Tweedledee but they can (and should) be distinguished. Psychological Science, 1, 46-54.
- Treisman, A. M., & Gelade, G. (1980). A feature-integration theory of attention. Cognitive Psychology, 12, 97-136.
- Pashler, H. (1994). Dual-task interference in simple tasks: Data and theory. Psychological Bulletin, 116, 220-244.
