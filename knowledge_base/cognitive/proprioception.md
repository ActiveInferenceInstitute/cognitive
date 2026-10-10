---
title: Proprioception
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - somatosensation
  - motor-control
  - embodiment
semantic_relations:
  - type: relates
    links:
      - "[[motor_control]]"
      - "[[sensorimotor_coordination]]"
      - "[[active_inference]]"
---

# Proprioception

## Overview
Proprioception is the sense of the position, movement and force of the body's own parts, independent of vision. It lets a person touch their nose with eyes closed or notice that an arm is raised. Sherrington introduced the term in the early twentieth century.

## Mechanism
The main receptors are muscle spindles, which signal muscle length and its rate of change through Ia and II afferents, and Golgi tendon organs, which signal tension through Ib afferents. Joint and cutaneous receptors, especially skin stretch, contribute. Spindle sensitivity is itself regulated by gamma motor neurons, so the sensor is controlled by the motor system and responses depend on the current motor command, not on muscle length alone.

Afferents project via the dorsal column pathways and spinocerebellar tracts to the cerebellum, somatosensory cortex (including area 3a and area 2) and posterior parietal cortex. Proprioceptive signals are combined with vision, vestibular input and efference copy to estimate limb state. Vibrating a muscle tendon excites spindles and produces illusory movement of the limb (Goodwin, McCloskey and Matthews), showing that perceived position is inferred from afferent signals.

Clinical loss of proprioception, as in large-fiber sensory neuropathy, leaves strength intact but severely impairs coordination, particularly without vision. Proprioception also underlies the sense of body position needed for [[body_schema]] updating.

## Role in Active Inference
Proprioception is central to the active inference account of movement. Adams, Shipp and Friston proposed that motor commands are not issued as such: descending signals carry proprioceptive predictions about the desired sensory consequences of movement, and spinal reflex arcs minimize the resulting proprioceptive prediction error by moving the muscles. Action is then perception's counterpart: instead of changing beliefs to fit the data, the body changes the data to fit the prediction.

For this to work, the precision of proprioceptive prediction error must be transiently reduced during movement initiation, which the framework proposes accounts for sensory attenuation. This is an influential theoretical account; it proposes a reinterpretation of the same circuits that classic motor control describes with forward models and efference copy, and the experimental data do not yet single out one over the other.

## Related concepts
- [[motor_control]] — movement generation and control schemes
- [[sensorimotor_coordination]] — integration of sensing and acting
- [[body_schema]] — body representation built from proprioceptive input
- [[interoception]] — complementary sense of internal state
- [[embodied_cognition]] — bodily basis of cognition
- [[precision_weighting]] — proposed mechanism of sensory attenuation

## References
- Adams, R. A., Shipp, S., & Friston, K. J. (2013). Predictions not commands: active inference in the motor system. Brain Structure and Function.
- Goodwin, G. M., McCloskey, D. I., & Matthews, P. B. C. (1972). The contribution of muscle afferents to kinaesthesia shown by vibration induced illusions of movement and by the effects of paralysing joint afferents. Brain.
