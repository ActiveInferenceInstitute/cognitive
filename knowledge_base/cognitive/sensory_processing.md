---
title: Sensory Processing
type: concept
status: stable
created: 2026-02-06
updated: 2026-02-07
tags:
  - sensory-processing
  - perception
  - predictive-coding
  - precision
  - neural-computation
semantic_relations:
  - type: relates
    links:
      - [[predictive_coding]]
      - [[precision_weighting]]
      - [[attention_mechanisms]]
      - [[perceptual_inference]]
      - [[visual_perception]]
---

# Sensory Processing

## Overview

Sensory processing is the neurobiological foundation of perception — transforming physical stimuli into neural representations. Under the predictive processing framework, sensory processing is not passive feature extraction but an active process of comparing top-down predictions against bottom-up sensory evidence, mediated by precision weighting.

## Predictive Coding in Sensory Systems

### Hierarchical Processing

```math
\begin{aligned}
& \varepsilon_l = x_l - f(\mu_{l+1}) \quad \text{(prediction error at level } l\text{)} \\
& \dot{\mu}_l = D\mu_l + \kappa_l (\varepsilon_l - \varepsilon_{l+1}) \quad \text{(belief update)} \\
& \text{where } \kappa_l = \Pi_l \text{ (precision weighting)}
\end{aligned}
```

### Sensory Modalities

| Modality | Stimulus | Receptor | Cortical Area | Prediction Example |
| --- | --- | --- | --- | --- |
| Vision | Light (380-740nm) | Retinal photoreceptors | V1, V2, V4 | Expected edges, textures |
| Audition | Sound (20Hz-20kHz) | Cochlear hair cells | A1, Belt | Expected pitch pattern |
| Somatosensory | Pressure, temperature | Mechanoreceptors | S1, S2 | Expected tactile contact |
| Olfaction | Volatile molecules | Olfactory epithelium | Piriform cortex | Expected odor |
| Gustation | Chemical compounds | Taste buds | Insula | Expected flavor |

### Multi-Sensory Integration

When multiple modalities provide evidence about the same state:

```math
q(s) \propto \prod_m p(o_m | s)^{\pi_m} \cdot p(s)
```

where $\pi_m$ is the precision of modality $m$. Higher-precision modalities dominate the posterior — this explains visual capture, the ventriloquist effect, and other cross-modal phenomena.

## Precision in Sensory Processing

Precision (inverse variance) plays a critical role:

```python
def precision_weighted_integration(observations, precisions, prior, prior_precision):
    """Multi-modal sensory integration with precision weighting."""
    total_precision = prior_precision + sum(precisions)
    posterior_mean = (prior * prior_precision + sum(
        o * p for o, p in zip(observations, precisions))) / total_precision
    posterior_precision = total_precision
    return posterior_mean, posterior_precision
```

### Gain Control as Precision

Neural gain control mechanisms implement precision weighting:
- **Contrast gain**: Early visual processing
- **Response gain**: Attentional modulation
- **Synaptic gain**: Neuromodulatory control (dopamine, acetylcholine, noradrenaline)

## Related Topics

- [[predictive_coding]] — Predictive coding framework
- [[precision_weighting]] — Precision mechanisms
- [[visual_perception]] — Visual perception
- [[perception_processing]] — Perceptual inference
- [[attention_mechanisms]] — Attention
- [[cross_modal_perception]] — Cross-modal integration
- [[multisensory_integration]] — Multisensory integration
