---
title: Optogenetics
description: A technique that uses light-sensitive proteins expressed in genetically specified cells to control or record neural activity, covering its principles, applications, limitations, and use for testing theories of neural computation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neuroscience
  - methods
  - neural_circuits
semantic_relations:
  - type: relates
    links:
      - "[[neural_circuits]]"
      - "[[brain_stimulation]]"
      - "[[neural_coding]]"
      - "[[neuroscience]]"
      - "[[neural_plasticity]]"
aliases:
  - optogenetic_manipulation
wikidata: Q781492
---

# Optogenetics

## Overview
Optogenetics is a set of methods in which cells are made sensitive to light by expressing genes for light-responsive proteins, so that their activity can be controlled, and in some variants reported, with light. Its defining advantage is cell-type specificity combined with millisecond timing, which allows causal tests of the role of defined neuron populations in circuit function and behaviour. It complements electrical stimulation and pharmacology, which act less selectively (see [[brain_stimulation]]).

## Principles
The foundation is the microbial opsins, light-gated membrane proteins. Channelrhodopsin-2, a light-gated cation channel from green algae (Nagel et al., 2003), depolarises neurons when illuminated with blue light, and Boyden and colleagues (2005) showed that expressing it in mammalian neurons allows spikes to be triggered with millisecond precision. Other opsins, such as halorhodopsin and archaerhodopsin, pump ions to hyperpolarise cells and silence them. Specificity comes from delivering the opsin gene under a cell-type-specific promoter, through transgenic animals or viral vectors, and light is delivered through optical fibres or other implanted devices. Related tools use light-sensitive proteins as sensors of voltage or calcium.

## Applications and limits
- **Causal mapping**: activating or silencing identified populations while observing behaviour or other brain regions, as in studies of memory, reward and motor control (see [[neural_circuits]]).
- **Circuit dissection**: testing the connectivity between cell types by stimulating axon terminals in a downstream area.
- **Disease models and translational work**, including early exploration of vision restoration.

Limitations include heating and light scattering in tissue, the invasiveness of delivery, non-physiological synchrony of evoked activity, variable expression, and the difficulty of generalising from tightly controlled manipulations to natural activity patterns. Interpretation therefore requires controls and convergent evidence (Deisseroth, 2015).

## Relation to Active Inference
Optogenetics has no direct connection to the Active Inference formalism. It is a method for testing proposals about neural computation, including those of predictive processing, by perturbing particular cell types. For example, hypotheses that distinct cortical populations encode predictions and prediction errors, or that neuromodulators set precision, can in principle be tested by selective activation or silencing (see [[predictive_coding]], [[neural_coding]]). Such experiments exist for circuit-level predictions in sensory cortex, but a direct test of the theory's specific claims remains rare.

## Related concepts
- [[neural_circuits]] — the units of analysis for circuit dissection
- [[brain_stimulation]] — less selective electrical and magnetic methods
- [[neural_coding]] — representations that perturbation can test
- [[neural_plasticity]] — changes in connectivity probed with light
- [[neuroscience]] — the wider field

## References
- Nagel, G., Szellas, T., Huhn, W., Kateriya, S., Adeishvili, N., Berthold, P., Ollig, D., Hegemann, P., & Bamberg, E. (2003). Channelrhodopsin-2, a directly light-gated cation-selective membrane channel. Proceedings of the National Academy of Sciences, 100, 13940-13945.
- Boyden, E. S., Zhang, F., Bamberg, E., Nagel, G., & Deisseroth, K. (2005). Millisecond-timescale, genetically targeted optical control of neural activity. Nature Neuroscience, 8, 1263-1268.
- Deisseroth, K. (2015). Optogenetics: 10 years of microbial opsins in neuroscience. Nature Neuroscience, 18, 1213-1225.
