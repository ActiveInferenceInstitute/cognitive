---
title: Neuroprosthetics
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - neurotechnology
  - sensory-restoration
  - plasticity
semantic_relations:
  - type: relates
    links:
      - "[[brain_machine_interfaces]]"
      - "[[neural_plasticity]]"
      - "[[sensory_processing]]"
---

# Neuroprosthetics

## Overview
Neuroprosthetics are devices that interface with the nervous system to replace or augment a lost sensory, motor or cognitive function. They either stimulate neural tissue to supply information the body no longer provides, or record neural signals to drive an artificial effector, and some do both.

## Mechanism
Sensory prostheses deliver artificial input. The cochlear implant is the most widely used: an external processor splits sound into frequency bands and drives a small number of electrodes (on the order of 12 to 22) along the cochlea, exploiting its tonotopic layout to stimulate auditory nerve fibers. Retinal prostheses stimulate remaining retinal cells with electrode arrays; the epiretinal Argus II used 60 electrodes and gives coarse spatial vision such as light perception and large shapes. Cortical visual prostheses and intracortical microstimulation of somatosensory cortex are experimental routes for restoring vision and touch.

Motor prostheses use recorded activity to move a cursor, robotic limb or stimulated muscles, and are the output side of [[brain_machine_interfaces]]. Bidirectional devices pair them with sensory feedback, since control improves when touch or proprioceptive information is returned. Deep brain stimulation, used for movement disorders, is a related neuromodulation therapy and is not usually called a prosthesis in the sensory-restoration sense.

Performance is limited by the resolution of electrical stimulation, which activates broad neural populations rather than patterns resembling natural activity, and by tissue response, electrode longevity and power. Outcomes depend strongly on plasticity: users generally need weeks or months of experience before the artificial input becomes useful, and age at implantation matters for cochlear implants in congenitally deaf children.

## Role in Active Inference
A prosthesis changes the mapping from world to sensory signal. In generative-model terms, the brain must learn a new likelihood relating hidden causes (a voice, an object) to the degraded and unfamiliar patterns the device delivers, which is a form of [[perceptual_learning]] driven by prediction error. Prior experience and the precision assigned to the new channel affect how quickly it is integrated with other senses. This is a reasonable framing but remains largely theoretical; clinical outcomes are usually modeled with signal-processing and plasticity concepts without reference to the free-energy principle.

## Related concepts
- [[brain_machine_interfaces]] — decoding and control side
- [[auditory_processing]] — natural system a cochlear implant replaces
- [[neural_plasticity]] — adaptation to the device
- [[perceptual_learning]] — learning to use artificial input
- [[proprioception]] — feedback channel targeted by sensory prostheses
- [[body_schema]] — incorporation of a device into the body model
