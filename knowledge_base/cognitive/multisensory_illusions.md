---
title: Multisensory Illusions
description: Perceptual illusions that arise when signals from different senses conflict or are combined, including the McGurk, ventriloquist and rubber hand illusions, and their explanation by Bayesian inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - perception
  - multisensory
  - bayesian_inference
semantic_relations:
  - type: relates
    links:
      - "[[multisensory_integration]]"
      - "[[cross_modal_perception]]"
      - "[[cue_integration]]"
      - "[[body_ownership]]"
      - "[[perceptual_inference]]"
---

# Multisensory Illusions

## Overview
Multisensory illusions are perceptual experiences that differ from the stimulus in any single sense because information from another sense alters the percept. They occur when the brain combines cues that conflict, or that are close enough in space and time to be attributed to a common cause. They are useful because they expose the rules of [[multisensory_integration]], which normally operates unnoticed and in the service of accurate perception.

## Examples
- **McGurk effect.** A video of a talker producing one syllable, paired with the sound of a different syllable, can be heard as a third syllable (McGurk and MacDonald, 1976; see [[lip_reading]]).
- **Ventriloquist effect.** A sound is mislocated toward a synchronous visual event, because vision usually has finer spatial resolution than hearing.
- **Rubber hand illusion.** Stroking a visible artificial hand in synchrony with the hidden real hand produces the feeling that the artificial hand belongs to the body (Botvinick and Cohen, 1998; see [[body_ownership]]).
- **Sound-induced flash illusion.** A single flash accompanied by two beeps is often seen as two flashes (Shams and colleagues, 2000).

## Explanation
Observers often weight cues in proportion to their reliability, which resembles statistically optimal combination (see [[cue_integration]]). Illusions occur when this process is driven by an erroneous assumption of a shared source. Causal inference models make this explicit: the brain infers whether signals have one cause or several and combines them in proportion to the posterior probability of a common cause, which accounts for the dependence of illusions on spatial and temporal discrepancy (Körding and colleagues, 2007; see [[causal_inference]]).

## Relation to Active Inference
Multisensory illusions are among the clearer applications of Active Inference and predictive processing. The rubber hand illusion has been modelled as inference about body ownership in which a strong prior for a common cause of visual and tactile events is combined with precision-weighted prediction errors, so that proprioceptive evidence is down-weighted (see [[perceptual_inference]] and [[precision_weighting]]). Such models reproduce qualitative features of the illusions and generate testable predictions, although alternative accounts of ownership exist.

## Related concepts
- [[multisensory_integration]] — the process the illusions probe
- [[cross_modal_perception]] — influence of one sense on another
- [[cue_integration]] — reliability-weighted combination of cues
- [[body_ownership]] — the sense of owning a body part
- [[bayesian_brain]] — the framework for interpreting illusions as inference

## References
- McGurk, H., & MacDonald, J. (1976). Hearing lips and seeing voices. Nature, 264, 746-748.
- Botvinick, M., & Cohen, J. (1998). Rubber hands 'feel' touch that eyes see. Nature, 391, 756.
- Körding, K. P., Beierholm, U., Ma, W. J., Quartz, S., Tenenbaum, J. B., & Shams, L. (2007). Causal inference in multisensory perception. PLoS ONE, 2, e943.
