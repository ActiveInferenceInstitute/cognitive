---
title: Emotional Inertia
description: The degree to which a person's emotional state carries over from one moment to the next, usually quantified by autocorrelation, and its associations with low wellbeing and depression.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - emotion
  - dynamics
  - wellbeing
semantic_relations:
  - type: relates
    links:
      - "[[emotion_regulation]]"
      - "[[depression]]"
      - "[[emotional_processing_expanded]]"
      - "[[emotional_granularity]]"
      - "[[time_series_analysis]]"
---

# Emotional Inertia

## Overview
Emotional inertia is the resistance of a person's emotional state to change. A person with high inertia stays in a given mood after it has begun, so that today's feeling strongly predicts the next feeling, whereas a person with low inertia returns quickly to a baseline or shifts flexibly with circumstances. The concept belongs to the study of emotion dynamics, which examines how emotions fluctuate over time within individuals.

## Measurement
Inertia is estimated from repeated self-reports, usually collected several times a day by experience sampling. A first-order autoregressive model describes the emotion $x_t$ as
$$x_t = c + \phi\, x_{t-1} + \varepsilon_t,$$
where the autoregressive coefficient $\phi$ is the index of inertia. Values near zero mean little carry-over, and values near one mean strong carry-over. Analyses must consider the person's mean level and variability, the spacing of measurements, and measurement error, all of which can affect the estimate. Variants model inertia of positive and negative emotion separately, or use multilevel and network models.

## Findings
Suls, Green and Hillis (1998) used the term affective inertia in connection with neuroticism. Kuppens, Allen and Sheeber (2010) found in two naturalistic studies that higher inertia of positive and negative emotion was associated with low self-esteem and with depression. Later work has linked high inertia of negative affect to rumination and reduced responsiveness to context, and has suggested that inertia may predict later onset of depressive symptoms in some samples. Effect sizes are small to moderate, and results differ across studies and measurement schedules, so inertia is better regarded as one marker among several than as a diagnostic tool.

## Relation to Active Inference
Persistence of an affective state can be described in predictive terms as strong, slowly updated beliefs about one's state, so incoming evidence changes the belief little. Some theoretical work on affect in Active Inference treats emotional valence as relating to changes in expected precision, but a quantitative account of emotional inertia has not been developed. The connection is therefore an analogy.

## Related concepts
- [[emotion_regulation]] — processes that can reduce persistence of unwanted states
- [[depression]] — the condition most associated with high inertia
- [[emotional_processing_expanded]] — extended treatment of affective processing
- [[emotional_granularity]] — another trait-level feature of emotional experience
- [[time_series_analysis]] — the statistical framework for autocorrelation

## References
- Suls, J., Green, P., & Hillis, S. (1998). Emotional reactivity to everyday problems, affective inertia, and neuroticism. Personality and Social Psychology Bulletin, 24, 127-136.
- Kuppens, P., Allen, N. B., & Sheeber, L. B. (2010). Emotional inertia and psychological maladjustment. Psychological Science, 21, 984-991.
- Kuppens, P., & Verduyn, P. (2017). Emotion dynamics. Current Opinion in Psychology, 17, 22-26.
