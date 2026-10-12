---
title: Reaction Time
description: The interval between a stimulus and the response to it, used as a measure of mental processing, covering Donders' method, Hick's law, response-time distributions and evidence-accumulation models.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - mental_chronometry
  - response_time
  - cognitive_measurement
semantic_relations:
  - type: relates
    links:
      - "[[processing_speed]]"
      - "[[evidence_accumulation]]"
      - "[[decision_making]]"
      - "[[psychophysics]]"
      - "[[language_processing]]"
wikidata: Q667379
---

# Reaction Time

## Overview
Reaction time, or response time, is the interval between the onset of a stimulus and the start of a response, such as a button press or an eye movement. It is among the oldest and most widely used behavioural measures in psychology, based on the assumption that longer times reflect more or longer processing. In simple reaction-time tasks, one stimulus requires one response. In choice tasks, different stimuli require different responses, and in go/no-go tasks the response is made only to some stimuli.

## Mental chronometry
Donders (1868) proposed the subtraction method: the time for a processing stage is estimated by subtracting the time of a simpler task from that of a task that adds the stage. The assumption that stages can be inserted without changing others is questionable. Sternberg (1969) developed additive-factor logic, in which factors that influence separate stages have additive effects on mean time, while factors that influence the same stage interact. In language research, reaction times in tasks such as lexical decision are used to infer how readily words are accessed ([[language_processing]]).

## Regularities
- **Hick's law.** In choice tasks, mean time increases with the amount of information to be processed, often written $RT=a+b\log_2(n+1)$ for $n$ equally likely alternatives (Hick, 1952).
- **Speed-accuracy trade-off.** Responding faster tends to increase errors, so time and accuracy must be interpreted together.
- **Distribution shape.** Times are right-skewed, with a long tail of slow responses, so the mean can mislead and distributional analyses are common.

## Models
Sequential-sampling models, such as the drift-diffusion model of Ratcliff (1978), explain choice accuracy and the full time distribution together. In them, noisy evidence accumulates until it reaches a threshold, and parameters for the rate of accumulation, the boundary separation and non-decision time can be estimated from data ([[evidence_accumulation]]).

## Relation to Active Inference
The relation is indirect. Evidence accumulation can be viewed as the progressive updating of beliefs about the correct response, and the response is made when confidence reaches a criterion, which resembles belief updating in Active Inference. Models in that framework can generate simulated response times from the number of belief-updating iterations or from the dynamics of simulated neural responses, but these are modelling choices, and the quantitative fit to human reaction-time distributions is less established than for the standard diffusion models ([[decision_making]]).

## Related concepts
- [[processing_speed]] — the speed of mental operations, often indexed by reaction time
- [[evidence_accumulation]] — models of how decisions develop over time
- [[decision_making]] — the choice processes that response times help to study
- [[psychophysics]] — a field that uses response times alongside accuracy
- [[language_processing]] — a domain where response-time tasks reveal access to words

## References
- Hick, W. E. (1952). On the rate of gain of information. Quarterly Journal of Experimental Psychology, 4, 11-26.
- Sternberg, S. (1969). The discovery of processing stages: Extensions of Donders' method. Acta Psychologica, 30, 276-315.
- Ratcliff, R. (1978). A theory of memory retrieval. Psychological Review, 85, 59-108.
- Luce, R. D. (1986). Response Times: Their Role in Inferring Elementary Mental Organization. Oxford University Press.
- Donders, F. C. (1868). Die Schnelligkeit psychischer Processe. Archiv für Anatomie, Physiologie und wissenschaftliche Medicin, 657-681. English translation: Donders, F. C. (1969). On the speed of mental processes (W. G. Koster, Trans.). Acta Psychologica, 30, 412-431.
