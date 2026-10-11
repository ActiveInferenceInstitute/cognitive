---
title: Corollary Discharge
description: Internal signals sent from motor to sensory systems that announce self-generated movement, enabling stable perception, sensory attenuation and distinction of self from external causes.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - sensorimotor
  - prediction
  - agency
semantic_relations:
  - type: relates
    links:
      - "[[efference_copy]]"
      - "[[forward_models]]"
      - "[[motor_control]]"
      - "[[sensorimotor_coordination]]"
      - "[[prediction_error]]"
---

# Corollary Discharge

## Overview
A corollary discharge is a signal generated within the motor system that is sent to sensory processing areas to inform them about an impending or ongoing self-generated movement. Sperry (1950) introduced the term to explain why a fish with inverted vision circled continuously, as if the visual motion produced by its own movement had been misinterpreted. The signal allows the nervous system to treat sensory changes caused by its own actions differently from those caused by the world.

## Functions and examples
Crapse and Sommer (2008) distinguish lower-level functions, such as sensory filtering and cancelling of self-generated input, from higher-level functions, such as stabilising perception across movements and supporting motor learning and planning. Familiar examples include the following.
- **Saccades**: the visual world appears stable despite abrupt eye movements. Corollary signals contribute both to suppressing visual input during saccades and to remapping receptive fields in advance of a movement; a pathway for such a signal in the primate brain, from the superior colliculus to frontal cortex via the thalamus, was described by Sommer and Wurtz (2002).
- **Self-touch**: sensations produced by one's own touch are rated as less intense than identical touches produced by another person or machine, and activity in somatosensory areas is correspondingly reduced (Blakemore, Wolpert and Frith, 1998).
- **Animal signalling**: a calling cricket suppresses its own auditory neurons during its chirps, protecting hearing sensitivity.

## Terminology
Corollary discharge and [[efference_copy]] are often used interchangeably. One convention reserves "efference copy" for a literal copy of a motor command and uses "corollary discharge" for the wider class of signals that notify sensory regions, whatever their origin. Another convention reverses the emphasis, so the usage should be checked in each source.

## Relation to Active Inference
Active Inference reinterprets the same phenomena without a separate command copy. Descending signals are proposed to be predictions of proprioceptive input that spinal reflex arcs fulfil, and sensory attenuation is explained by transiently reduced precision of the sensory evidence that would otherwise report a mismatch (Brown et al., 2013). Corollary discharge is then the part of the generative model that predicts the sensory consequences of action. This is an active theoretical programme, supported by simulations of sensory attenuation and the force-matching illusion, and it offers a different account of the same data rather than a distinct experimental result.

## Related concepts
- [[efference_copy]] — the closely related copy of the motor command
- [[forward_models]] — the predictors that use such signals
- [[motor_control]] — the system producing the signal
- [[sensorimotor_coordination]] — coupling of movement and sensation
- [[prediction_error]] — the quantity attenuated for self-generated input

## References
- Sperry, R. W. (1950). Neural basis of the spontaneous optokinetic response produced by visual inversion. Journal of Comparative and Physiological Psychology, 43, 482-489.
- Blakemore, S.-J., Wolpert, D. M., & Frith, C. D. (1998). Central cancellation of self-produced tickle sensation. Nature Neuroscience, 1, 635-640.
- Sommer, M. A., & Wurtz, R. H. (2002). A pathway in primate brain for internal monitoring of movements. Science, 296, 1480-1482.
- Crapse, T. B., & Sommer, M. A. (2008). Corollary discharge across the animal kingdom. Nature Reviews Neuroscience, 9, 587-600.
- Brown, H., Adams, R. A., Parees, I., Edwards, M., & Friston, K. (2013). Active inference, sensory attenuation and illusions. Cognitive Processing, 14, 411-427.
