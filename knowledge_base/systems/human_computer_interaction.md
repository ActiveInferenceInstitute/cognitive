---
title: Human-Computer Interaction
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - systems
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/systems/spatial_computing]]"
      - "[[knowledge_base/cognitive/user_modeling]]"
      - "[[knowledge_base/cognitive/attention_mechanisms]]"
---

# Human-Computer Interaction

## Overview

Human-computer interaction (HCI) studies how people use computing systems and how to design systems that fit human perception, memory, motor control and goals. It combines empirical methods (usability tests, controlled experiments, field studies) with models that predict performance. Active Inference enters as a framework for describing both parties as agents who predict one another. The performance models in HCI are established; the Active Inference reading of interaction is a research proposal and has seen limited application to real interface studies.

## Established Foundations

**Performance laws and models.** Fitts's law predicts pointing time from target distance and width. The GOMS family predicts expert task time from goals, operators, methods and selection rules. Hick's law relates choice time to the number of options.

**Norman's gulfs.** Users must bridge a gulf of execution (how do I do what I want?) and a gulf of evaluation (what state is the system in?). Feedback, affordances and consistent conventions narrow them.

**Cognitive load.** Working memory is limited, so interfaces that demand holding many items or switching context are error-prone ([[knowledge_base/cognitive/working_memory]]).

**Attention and interruption.** Notifications compete for attention; research on interruptibility asks when a prompt is least costly ([[knowledge_base/cognitive/attention_mechanisms]]).

## Active Inference Reading

A user acting on an interface carries expectations about what each action will do. A smooth interaction is one where those predictions are fulfilled; surprising behaviour, such as a menu that rearranges itself, is a prediction error that costs attention and may cause mistakes. This gives a principled account of why consistency and learnability matter, and of why adaptive layouts can hurt even when they save pointing distance on average. The same framing applies to the system: an adaptive system infers the user's goal and state and picks displays or prompts by expected free energy, which weighs the value of acting on its best guess against asking ([[knowledge_base/cognitive/user_modeling]]). Attention-aware behaviour follows from tracking the precision of the user's attention as a hidden state.

The practical limits are the same as for user modeling: identifiability of goals from sparse input, cold start, and the need for transparency and consent.

## In This Repository

Related pages: [[knowledge_base/cognitive/user_modeling]], [[knowledge_base/systems/spatial_computing]], [[knowledge_base/systems/augmented_reality]], [[knowledge_base/systems/virtual_reality]] and [[knowledge_base/systems/mixed_reality]], which cover immersive interfaces. No interface, user-study tool or interaction simulator is in the code. What the code offers is the inference machinery in `code/tools/src/models/active_inference`, and plotting helpers for beliefs and free energy in `code/tools/src/visualization` that could display model state to a designer.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/systems/spatial_computing]]
