---
title: Body Schema
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - embodiment
  - self-representation
  - multisensory
semantic_relations:
  - type: relates
    links:
      - "[[embodied_cognition]]"
      - "[[proprioception]]"
      - "[[self_awareness]]"
wikidata: Q907574
---

# Body Schema

## Overview
The body schema is a sensorimotor representation of the body's current configuration, size and capabilities that the brain uses to guide action, usually without awareness. It is typically contrasted with body image, the conscious perception, beliefs and attitudes about one's body. The distinction is used by Gallagher, though terminology varies across authors and the boundary is debated.

## Mechanism
The concept goes back to Head and Holmes (1911), who proposed a postural model of the body that is continuously updated by afferent signals. The schema is built from proprioception, touch, vision, vestibular input and efference copy, and is thought to depend on posterior parietal cortex, premotor cortex and the cerebellum.

It is plastic. Tool use extends the effective reach of the schema: neurons in monkey parietal cortex with hand-centered visual receptive fields expanded after the animal used a rake (reviewed by Maravita and Iriki). In the rubber hand illusion, synchronous touch on a visible fake hand and the hidden real hand leads people to feel ownership of the rubber hand and shifts perceived hand position (Botvinick and Cohen). Phantom limbs after amputation, and the persistence of schema information for missing limbs, illustrate its partial independence from the actual body. Neglect and somatoparaphrenia after parietal damage show specific disorders of body representation.

Computational accounts treat the schema as an internal model, a kinematic and dynamic description used by forward models for planning and predicting movement consequences.

## Role in Active Inference
In predictive-processing accounts, the body schema is a set of prior beliefs about the body that generates proprioceptive and tactile predictions. Multisensory conflict, as in the rubber hand illusion, is then resolved by inferring the common cause best explaining visual, tactile and proprioceptive signals; ownership is high when the model with a single body part explains them best. Apps and Tsakiris applied this to self-recognition, and Samad and colleagues modeled body ownership as Bayesian causal inference. Because the same schema supplies the proprioceptive predictions that [[active_inference]] proposes drive movement, it links perception of the body to action.

These are modelling proposals. Several phenomena, such as the persistence of phantom limbs, have been explained by plasticity and efference-copy accounts without reference to free energy.

## Related concepts
- [[proprioception]] — principal sensory input for the schema
- [[embodied_cognition]] — role of the body in cognition
- [[self_awareness]] — conscious self-representation
- [[multisensory_integration]] — combination of cues in ownership illusions
- [[motor_control]] — consumer of the schema in action planning
- [[neuroprosthetics]] — incorporation of devices into the schema

## References
- Head, H., & Holmes, G. (1911). Sensory disturbances from cerebral lesions. Brain.
- Botvinick, M., & Cohen, J. (1998). Rubber hands 'feel' touch that eyes see. Nature.
- Maravita, A., & Iriki, A. (2004). Tools for the body (schema). Trends in Cognitive Sciences.
- Apps, M. A. J., & Tsakiris, M. (2014). The free-energy self: a predictive coding account of self-recognition. Neuroscience & Biobehavioral Reviews.
