---
title: Motor Prosthetics
description: Devices that restore movement or its effects by reading motor commands from the nervous system or muscles, including intracortical brain-machine interfaces, with notes on decoding and sensory feedback.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neuroprosthetics
  - motor_control
  - brain_machine_interfaces
semantic_relations:
  - type: relates
    links:
      - "[[brain_machine_interfaces]]"
      - "[[neuroprosthetics]]"
      - "[[motor_cortex]]"
      - "[[neural_decoding]]"
      - "[[population_vector]]"
---

# Motor Prosthetics

## Overview
A motor prosthetic restores the output side of the nervous system: it takes a signal that expresses movement intent and uses it to drive an effector such as a cursor, a robotic arm or the person's own paralysed muscles. The signal may come from residual muscles, as in myoelectric limbs, or directly from the brain, as in intracortical brain-machine interfaces (see [[brain_machine_interfaces]]). Sensory prosthetics such as cochlear implants restore input instead and are treated separately.

## Decoding movement intent
In intracortical systems, arrays of microelectrodes in motor or premotor cortex record the activity of many neurons. Neurons in primary motor cortex show broad tuning to movement direction, so the combined activity can be read out using a population vector or a statistical filter (see [[population_vector]], [[neural_decoding]]). Wessberg et al. (2000) showed that simultaneously recorded cortical ensembles could be used to reconstruct, and in real time to control, a robotic arm in monkeys. Hochberg et al. (2006) reported that a person with tetraplegia could control a computer cursor and simple devices through such an array, and Hochberg et al. (2012) reported reach and grasp with a robotic arm in two participants. Typical challenges include the decline of recorded signal quality over months, the need for recalibration, and surgical risk. Users learn to improve control through practice, so the decoder and the brain adapt to each other.

## Sensory feedback
Control without touch or proprioception is slow and imprecise, so a current research direction is to deliver artificial sensation, for example by stimulating somatosensory cortex or peripheral nerves. How naturalistic such sensation can become is an open question.

## Relation to Active Inference
Closed-loop prosthetic control is a natural setting for sensorimotor theories in which movement arises from predictions of proprioceptive consequences, not from explicit commands (Adams et al., 2013). This is an interpretive link: the engineering of decoders is mostly based on regression and filtering, and the formalism has not been shown to improve prosthetic performance.

## Related concepts
- [[brain_machine_interfaces]] — the broader class of devices
- [[neuroprosthetics]] — restoring function with implants
- [[motor_cortex]] — the usual recording site
- [[electrode_arrays]] — the recording hardware
- [[multi_unit]] — a signal type often used for decoding

## References
- Wessberg, J., Stambaugh, C. R., Kralik, J. D., Beck, P. D., Laubach, M., Chapin, J. K., ... Nicolelis, M. A. L. (2000). Real-time prediction of hand trajectory by ensembles of cortical neurons in primates. Nature, 408, 361-365.
- Hochberg, L. R., Serruya, M. D., Friehs, G. M., Mukand, J. A., Saleh, M., Caplan, A. H., ... Donoghue, J. P. (2006). Neuronal ensemble control of prosthetic devices by a human with tetraplegia. Nature, 442, 164-171.
- Hochberg, L. R., Bacher, D., Jarosiewicz, B., Masse, N. Y., Simeral, J. D., Vogel, J., ... Donoghue, J. P. (2012). Reach and grasp by people with tetraplegia using a neurally controlled robotic arm. Nature, 485, 372-375.
- Adams, R. A., Shipp, S., & Friston, K. J. (2013). Predictions not commands: Active inference in the motor system. Brain Structure and Function, 218, 611-643.
