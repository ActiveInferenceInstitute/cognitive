---
title: Personality
description: Personality refers to stable individual differences in patterns of thought, emotion, and behaviour, summarized by trait models such as the Big Five, with tentative computational links to Active Inference priors.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - individual-differences
  - traits
  - emotion
semantic_relations:
  - type: relates
    links:
      - "[[individual_differences]]"
      - "[[emotional_processing]]"
      - "[[social_cognition]]"
---

# Personality

## Overview
Personality is the relatively stable organization of an individual's characteristic patterns of thinking, feeling, and behaving. The dominant scientific approach is trait psychology, which describes people by position on continuous dimensions and studies their stability, origins, and consequences. The sense used here is trait personality, as opposed to clinical personality disorders or lay character judgments.

## Definition / Formalism
**Big Five model.** Factor analyses of trait descriptions and questionnaires recover five broad dimensions: openness to experience, conscientiousness, extraversion, agreeableness, and neuroticism (emotional stability), each decomposed into narrower facets. Alternatives include the six-factor HEXACO model, which adds honesty-humility.

**Properties.** Rank-order stability increases across the lifespan, while mean levels shift modestly (for example, conscientiousness tends to rise with age). Twin and adoption studies attribute a moderate share of trait variance to genetic differences, typically cited at around 40 percent, with the remainder mostly non-shared environment and measurement error (Bouchard and Loehlin, 2001). Traits predict outcomes such as health behaviour, work performance, and relationship satisfaction, with small to moderate effect sizes.

**Critiques.** The person-situation debate showed that traits predict behaviour better when aggregated across situations than in a single instance. Self-report is open to response bias, and trait labels describe rather than explain behaviour. Cross-cultural replication of the five-factor structure is imperfect.

**Process accounts.** Cybernetic Big Five Theory (DeYoung, 2015) treats traits as stable parameters of goal-directed, feedback-controlled systems, for instance linking neuroticism to sensitivity to threat and error, and extraversion to sensitivity to reward. Neuroticism and related emotional traits link to the processing described in [[emotional_processing]] and [[anxiety]].

## Role in Active Inference
Active Inference supplies a vocabulary in which trait differences could correspond to stable differences in model parameters: prior beliefs about the world's volatility and controllability, precision assigned to prediction errors, the strength of prior preferences, and the balance of exploration and exploitation ([[prior_beliefs]], [[precision_mechanisms]]). For example, high neuroticism might be formalized as strong prior expectation of aversive outcomes. This is a conceptual mapping and a programme for computational phenotyping, not an established result; no accepted parameter set corresponds to the Big Five, and the approach competes with reinforcement-learning-based parameterizations.

## Related concepts
- [[individual_differences]] — wider study of variation among people
- [[emotional_processing]] — affective processes tied to trait dimensions
- [[social_cognition]] — traits influence social perception and behaviour
- [[developmental_psychology]] — how personality emerges and changes
- [[computational_psychiatry]] — parameter-based accounts of individual variation
- [[exploration_exploitation]] — trade-off whose bias may vary between individuals

## References
- Bouchard, T. J., & Loehlin, J. C. (2001). Genes, evolution, and personality. *Behavior Genetics*, 31, 243-273.
- DeYoung, C. G. (2015). Cybernetic Big Five Theory. *Journal of Research in Personality*, 56, 33-58.
