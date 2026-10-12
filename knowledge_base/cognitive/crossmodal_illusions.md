---
title: Crossmodal Illusions
description: Perceptual illusions in which information from one sense alters or creates a percept in another, including the McGurk, ventriloquist, rubber hand and sound-induced flash effects, and their Bayesian causal inference account.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - multisensory_perception
  - illusions
  - bayesian_inference
semantic_relations:
  - type: relates
    links:
      - "[[multisensory_illusions]]"
      - "[[multisensory_integration]]"
      - "[[cue_integration]]"
      - "[[cross_modal_perception]]"
      - "[[lip_reading]]"
---

# Crossmodal Illusions

## Overview
Crossmodal illusions are perceptual experiences in which input from one sensory modality changes or produces a percept in another. They arise because the senses are not processed in isolation: signals that are likely to share a cause are combined, and when the signals conflict the combined percept can differ from either source. They are therefore useful for revealing the rules of multisensory integration.

## Classic examples
- **McGurk effect.** A video of a face articulating one syllable paired with a different spoken syllable often yields a third percept, for example visual "ga" and auditory "ba" heard as "da" (McGurk & MacDonald, 1976).
- **Ventriloquist effect.** The apparent location of a sound shifts toward a concurrent visual event. Alais and Burr (2004) showed that the shift follows the relative reliabilities of the two cues, close to the weighting an optimal observer would use.
- **Rubber hand illusion.** Synchronous stroking of a visible artificial hand and the participant's hidden hand can lead to the experience that the artificial hand belongs to the body (Botvinick & Cohen, 1998).
- **Sound-induced flash illusion.** A single flash accompanied by two brief sounds is often reported as two flashes.

## Causal inference account
For two cues with Gaussian noise of variances $\sigma_A^2$ and $\sigma_V^2$, the reliability-weighted estimate is $\hat{s}=(w_A x_A + w_V x_V)/(w_A+w_V)$ with $w_i = 1/\sigma_i^2$. Körding and colleagues (2007) extended this by letting the observer infer whether the signals share a single cause. If a common cause is probable, the cues are fused. If not, they are treated separately. Illusions are strongest when spatial or temporal discrepancy is small enough for a common cause to remain plausible.

## Role in Active Inference
Crossmodal illusions are naturally cast as inference under a generative model with a prior that coincident signals share a cause. Apps and Tsakiris (2014) applied a free-energy account to self-recognition and to the rubber hand illusion, treating body ownership as inference over multisensory evidence. This is an active research programme, and competing explanations of ownership and speech perception also exist.

## Related concepts
- [[multisensory_illusions]] — closely related page on illusions across senses
- [[multisensory_integration]] — the process whose rules the illusions expose
- [[cue_integration]] — reliability-weighted combination of cues
- [[cross_modal_perception]] — the wider study of interactions between senses
- [[lip_reading]] — visual speech information underlying the McGurk effect

## References
- McGurk, H., & MacDonald, J. (1976). Hearing lips and seeing voices. Nature, 264, 746-748.
- Botvinick, M., & Cohen, J. (1998). Rubber hands "feel" touch that eyes see. Nature, 391, 756.
- Alais, D., & Burr, D. (2004). The ventriloquist effect results from near-optimal bimodal integration. Current Biology, 14, 257-262.
- Körding, K. P., Beierholm, U., Ma, W. J., Quartz, S., Tenenbaum, J. B., & Shams, L. (2007). Causal inference in multisensory perception. PLoS ONE, 2, e943.
- Apps, M. A. J., & Tsakiris, M. (2014). The free-energy self: A predictive coding account of self-recognition. Neuroscience & Biobehavioral Reviews, 41, 85-97.
