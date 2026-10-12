---
title: Cue Integration
description: Combining multiple noisy sensory cues about the same quantity, with reliability-weighted averaging, evidence for near-optimal human performance, causal inference, and the Active Inference interpretation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - perception
  - bayesian_inference
  - multisensory
semantic_relations:
  - type: relates
    links:
      - "[[bayesian_brain]]"
      - "[[multisensory_integration]]"
      - "[[precision_weighting]]"
      - "[[cross_modal_perception]]"
      - "[[depth_perception]]"
      - "[[gaussian_distributions]]"
---

# Cue Integration

## Overview
Cue integration is the combination of several sources of sensory evidence about one property of the world. Size can be estimated by sight and by touch, location by sound and by vision, and depth by stereo disparity, texture and motion. Each cue is noisy, and combining them can yield an estimate more reliable than any single cue.

## Optimal combination
Suppose each cue gives an unbiased estimate $\hat s_i$ with independent Gaussian noise of variance $\sigma_i^2$. The maximum-likelihood combined estimate is a weighted average
$$\hat s=\sum_i w_i \hat s_i,\qquad w_i=\frac{1/\sigma_i^2}{\sum_j 1/\sigma_j^2},$$
whose variance is $\sigma^2=\left(\sum_i 1/\sigma_i^2\right)^{-1}$, which is never larger than the smallest $\sigma_i^2$. The same result follows from Bayesian inference with a flat prior. With an informative prior, the prior acts as one more cue.

## Evidence and limits
Ernst and Banks (2002) showed that human judgements of object size from vision and touch matched these predictions: when visual noise was added, observers shifted weight toward touch, and bimodal precision exceeded that of either modality alone. Alais and Burr (2004) obtained a similar pattern for audiovisual localisation, explaining the ventriloquist effect as visual dominance when vision is the more reliable cue and sound dominance when blur makes vision unreliable. Integration is not always complete. If cues are too discrepant, observers behave as if they may have different causes. Körding and colleagues modelled this with Bayesian causal inference, which infers whether cues share a source before deciding how much to combine them.

## Relation to Active Inference
Under a Gaussian generative model, precision-weighted prediction errors reproduce the weights above, so cue integration is a standard worked example of Bayesian perception and of the role of precision in predictive processing. In that sense the connection is a derivation. Whether neural populations implement these computations by the proposed message-passing scheme is an empirical hypothesis, as is the claim that humans are optimal outside controlled laboratory tasks.

## Related concepts
- [[bayesian_brain]] — the hypothesis that perception approximates Bayesian inference
- [[multisensory_integration]] — the broader phenomenon across senses
- [[precision_weighting]] — reliability weighting in predictive processing
- [[cross_modal_perception]] — interactions between modalities
- [[depth_perception]] — a domain with many separable cues
- [[gaussian_distributions]] — the noise model behind the formulas

## References
- Ernst, M. O., & Banks, M. S. (2002). Humans integrate visual and haptic information in a statistically optimal fashion. Nature, 415, 429-433.
- Alais, D., & Burr, D. (2004). The ventriloquist effect results from near-optimal bimodal integration. Current Biology, 14, 257-262.
- Körding, K. P., Beierholm, U., Ma, W. J., Quartz, S., Tenenbaum, J. B., & Shams, L. (2007). Causal inference in multisensory perception. PLoS ONE, 2(9), e943.
