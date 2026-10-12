---
title: Response Conflict
description: Competition between incompatible response tendencies, how it is measured in interference tasks, how conflict monitoring detects it, and how it relates to policy competition in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - executive_function
  - cognitive_control
  - interference
semantic_relations:
  - type: relates
    links:
      - "[[conflict_monitoring]]"
      - "[[anterior_cingulate]]"
      - "[[inhibitory_control]]"
      - "[[cognitive_control]]"
      - "[[executive_functions]]"
---

# Response Conflict

## Overview
Response conflict arises when two or more activated responses cannot be executed together, for example when an automatic tendency and an instructed response point in different directions. It slows responses and raises error rates, and it is treated as a signal that control may be needed. The term is related to, but narrower than, conflict between goals or beliefs.

## Measuring conflict
- **Stroop task.** Naming the ink colour of an incongruent colour word is slower than for a congruent one, because word reading is an automatic competing response.
- **Flanker task.** Eriksen and Eriksen (1974) showed that flanking letters assigned to the opposite response slow identification of a central target.
- **Simon task.** Responses are slower when the stimulus location and the response side differ, even though location is irrelevant.
- **Go/no-go and stop-signal tasks.** Conflict arises between executing and withholding a prepotent response, which is often discussed under [[inhibitory_control]].

Typical findings include a congruency effect on reaction time and a sequence effect in which the effect is smaller after an incongruent trial than after a congruent one. The sequence effect has been interpreted as control adjustment, although low-level repetition and association confounds can contribute to it.

## Conflict monitoring
In the conflict-monitoring theory of Botvinick and colleagues (2001), the anterior cingulate cortex registers co-activation of competing responses and signals prefrontal regions to strengthen task-relevant processing. Evidence includes anterior cingulate activity on incongruent trials and a frontal midline error-related potential. Alternative accounts attribute the same activity to error likelihood or to effort and value computations, so the interpretation is still debated.

## Relation to Active Inference
In discrete Active Inference, competing responses correspond to policies with similar posterior probability, which appears as high policy entropy; precision on the policy prior then determines how decisively one wins. This is an interpretive mapping of conflict onto the framework, not a result tied to a specific model of the Stroop task.

## Related concepts
- [[conflict_monitoring]] — detection of response conflict
- [[anterior_cingulate]] — a region associated with monitoring
- [[cognitive_inhibition]] — suppression of the competing response
- [[task_switching]] — a setting where conflict between task sets arises
- [[automaticity]] — the source of many prepotent competing responses

## References
- Eriksen, B. A., & Eriksen, C. W. (1974). Effects of noise letters upon the identification of a target letter in a nonsearch task. Perception & Psychophysics, 16, 143-149.
- Botvinick, M. M., Braver, T. S., Barch, D. M., Carter, C. S., & Cohen, J. D. (2001). Conflict monitoring and cognitive control. Psychological Review, 108, 624-652.
- Stroop, J. R. (1935). Studies of interference in serial verbal reactions. Journal of Experimental Psychology, 18, 643-662.
