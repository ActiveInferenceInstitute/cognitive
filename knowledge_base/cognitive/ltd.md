---
title: Long-Term Depression (LTD)
description: Long-term depression is a lasting activity-dependent weakening of synaptic strength, complementing long-term potentiation, with calcium-dependent mechanisms and a role in learning and forgetting.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - plasticity
  - synapse
  - learning
semantic_relations:
  - type: relates
    links:
      - "[[synaptic_plasticity]]"
      - "[[hebbian_learning]]"
      - "[[neural_plasticity]]"
---

# Long-Term Depression (LTD)

## Overview
Long-term depression (LTD) is a persistent reduction in the efficacy of synaptic transmission, lasting from tens of minutes to much longer, induced by specific patterns of neural activity. It is the counterpart of long-term potentiation (LTP) and, together with it, allows synaptic weights to move in both directions. It is documented in the hippocampus, neocortex, striatum, and cerebellum. The cerebellar form, at parallel fibre to Purkinje cell synapses, was proposed by Ito as a basis of motor learning.

## Mechanism
In hippocampal area CA1, the best-studied form is NMDA-receptor-dependent LTD. Prolonged low-frequency stimulation (on the order of 1 Hz for several minutes) produces a modest, slow rise in postsynaptic calcium, which preferentially engages protein phosphatases such as calcineurin and PP1, in contrast to the larger, faster rise that activates kinases and produces LTP. The result is removal of AMPA receptors from the postsynaptic membrane, reducing the response to glutamate. A second form depends on metabotropic glutamate receptors (mGluR-LTD) and involves local protein synthesis. Endocannabinoid-mediated LTD is common in striatum.

LTD also appears as the timing-dependent half of spike-timing-dependent plasticity: when a presynaptic spike follows a postsynaptic one, the synapse is typically weakened. The calcium-level dependence is the basis of the BCM theory, in which a sliding threshold separates depression from potentiation (see [[metaplasticity]]).

Proposed functions include weakening of unused or uncorrelated inputs, refinement of receptive fields, prevention of runaway potentiation, memory updating and forgetting, and in the cerebellum, error-driven correction.

## Role in Active Inference
Active Inference needs learning rules that can both increase and decrease parameters. In continuous predictive-coding schemes, free-energy gradients give weight changes that are signed according to the prediction error, so depression arises when a prediction overshoots; this is formally Hebbian-like and has been proposed as the computational role of bidirectional plasticity. In count-based discrete models, depression corresponds to forgetting or decay of Dirichlet counts, which is an added assumption. The mapping from calcium-dependent LTD to these update rules is a modelling hypothesis, not a demonstrated identity.

## Related concepts
- [[synaptic_plasticity]] — the broader family of synaptic change including LTP
- [[hebbian_learning]] — correlation-based rule that LTD complements
- [[metaplasticity]] — threshold shifts that govern LTP versus LTD
- [[homeostatic_plasticity]] — stabilizing adjustments of overall synaptic strength
- [[memory_systems]] — systems whose storage depends on synaptic change
- [[motor_learning]] — cerebellar LTD proposed as a learning mechanism
- [[neural_plasticity]] — wider category of experience-dependent change
