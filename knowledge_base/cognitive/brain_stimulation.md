---
title: Brain Stimulation
description: Methods that modulate neural activity directly, including transcranial magnetic and electrical stimulation and deep brain stimulation, their proposed mechanisms, and their use as causal probes of cognition.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neuroscience
  - neuromodulation
  - methods
semantic_relations:
  - type: relates
    links:
      - "[[neural_plasticity]]"
      - "[[cognitive_neuroscience]]"
      - "[[brain_machine_interfaces]]"
aliases:
  - neural_stimulation
---

# Brain Stimulation

## Overview
Brain stimulation covers techniques that alter neural activity by applying electrical or magnetic fields, in contrast with recording methods that only observe it. In cognitive neuroscience they serve mainly as causal tools: if perturbing a region changes performance on a task, the region contributes to it. Clinical uses exist for some conditions, and their efficacy is a matter for clinical research; this page describes the methods and does not make treatment claims.

## Methods
- **Transcranial magnetic stimulation (TMS)**: a brief magnetic pulse from a coil induces currents in cortex beneath it, introduced for the human motor cortex by Barker, Jalinous and Freeston (1985). Single pulses or short trains can transiently disrupt local processing (a "virtual lesion"), while patterned protocols such as theta burst (Huang et al., 2005) produce effects lasting beyond the stimulation period (Hallett, 2007).
- **Transcranial direct current stimulation (tDCS)**: weak constant current shifts the excitability of neurons, with polarity-dependent effects on motor-evoked potentials (Nitsche and Paulus, 2000). Effects on cognition are generally small and variable across studies and individuals.
- **Deep brain stimulation (DBS)**: implanted electrodes deliver pulses to subcortical targets and are used clinically in some movement disorders; the mechanism is debated, with proposals involving local inhibition, axonal activation and altered network oscillations.
- **Others**: transcranial alternating current stimulation targets oscillatory entrainment; vagus-nerve stimulation acts via brainstem neuromodulatory systems.

Stimulation effects depend on state, current geometry and anatomy, which complicates inference from a behavioural change to a function of the stimulated region.

## Role in Active Inference
There is no dedicated Active Inference theory of stimulation; the link is indirect. Stimulation can be treated as an exogenous perturbation to neural states or synaptic gains, and a generative-model account makes predictions about the consequences, for example that altering the gain of prediction-error populations should change the effective precision of the corresponding inputs (see [[precision_weighting]]). Plasticity-inducing protocols connect to learning rules (see [[synaptic_plasticity]], [[hebbian_learning]]). Such predictions are mostly hypothetical, and their testing is an active, early-stage area of research.

## Related concepts
- [[memory_systems]] — stimulation is used to probe memory circuits
- [[language_processing]] — TMS has been used to test regional contributions to language
- [[neuroprosthetics]] — related approach to interfacing with neural tissue
- [[cholinergic_system]] — neuromodulatory systems that stimulation may engage

## References
- Barker, A. T., Jalinous, R., & Freeston, I. L. (1985). Non-invasive magnetic stimulation of human motor cortex. The Lancet, 325, 1106-1107.
- Nitsche, M. A., & Paulus, W. (2000). Excitability changes induced in the human motor cortex by weak transcranial direct current stimulation. Journal of Physiology, 527, 633-639.
- Huang, Y.-Z., et al. (2005). Theta burst stimulation of the human motor cortex. Neuron, 45, 201-206.
- Hallett, M. (2007). Transcranial magnetic stimulation: a primer. Neuron, 55, 187-199.
