---
title: Mathematical Psychology
description: Mathematical psychology builds formal models of perception, memory, learning, and choice, and this page covers its main model families and their relation to Active Inference.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - modeling
  - psychophysics
  - choice
semantic_relations:
  - type: relates
    links:
      - "[[psychophysics]]"
      - "[[signal_detection]]"
      - "[[computational_modeling]]"
wikidata: Q1908389
---

# Mathematical Psychology

## Overview
Mathematical psychology is the subfield that expresses psychological theories as explicit mathematical or computational models, derives quantitative predictions, and tests them against behavioural data. It overlaps with cognitive modelling and computational neuroscience but is distinguished by an emphasis on axiomatic or process models of behaviour, rigorous model comparison, and measurement theory. It supplies much of the quantitative language used in the perception and binding pages of this knowledge base.

## Core model families
- **Psychophysics and detection**: threshold and scaling laws, and signal detection theory, which separates sensitivity from response bias ([[psychophysics]], [[signal_detection]]).
- **Choice models**: Luce's choice axiom states that the probability of choosing option $a$ from a set depends on its strength relative to the others, $P(a) = v(a)/\sum_b v(b)$. Setting $v = e^{\beta u}$ gives the softmax rule ([[softmax_function]]).
- **Sequential sampling**: the diffusion model and race models represent decisions as noisy evidence accumulation to a bound, jointly explaining response time distributions and accuracy ([[evidence_accumulation]]).
- **Learning models**: stimulus-sampling theory, the Bush-Mosteller linear operator model, and error-driven reinforcement-learning rules ([[reinforcement_learning]]).
- **Memory and categorization**: exemplar and prototype models, and global matching models of recognition.
- **Decision under risk**: expected utility and its descriptive alternatives ([[utility_theory]]).
- **Measurement theory**: conditions under which psychological attributes can be assigned numbers and scales.

Methodological standards include fitting by maximum likelihood or Bayesian estimation, penalizing flexibility through information criteria or Bayes factors ([[model_comparison]]), parameter recovery from simulated data, and checking whether a model is identifiable at all.

## Role in Active Inference
Active Inference can be read as one more formal model class in this tradition, with the distinctive feature that perception, learning, and choice share a single objective. Its policy selection rule, a softmax over negative expected free energy scaled by a precision, is a Luce-type choice rule with a principled utility. Evidence accumulation appears as gradient descent on free energy in time-evolving beliefs. The main practical contribution is a generative-model fitting pipeline: simulated behaviour is fitted to participant data and models are compared through free-energy-based evidence bounds. Whether it outperforms specialized models such as drift diffusion on a given task is an empirical question, and identifiability of its parameters needs the same scrutiny applied to any model here.

## Related concepts
- [[computational_modeling]] — wider practice of formal modelling
- [[psychophysics]] — historical origin of quantitative psychology
- [[signal_detection]] — framework separating sensitivity and bias
- [[evidence_accumulation]] — sequential sampling accounts of decisions
- [[model_comparison]] — selecting among competing formal models
- [[softmax_function]] — the choice rule shared with Luce's axiom

## References
- Luce, R. D. (1959). *Individual Choice Behavior: A Theoretical Analysis*. Wiley.
- Green, D. M., & Swets, J. A. (1966). *Signal Detection Theory and Psychophysics*. Wiley.
- Ratcliff, R. (1978). A theory of memory retrieval. *Psychological Review*, 85, 59-108.
