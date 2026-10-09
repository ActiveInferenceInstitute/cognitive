---
title: Homeostatic Plasticity
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - plasticity
  - stability
  - synapse
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[synaptic_plasticity]]"
      - "[[stability_plasticity]]"
---

# Homeostatic Plasticity

## Overview
Homeostatic plasticity refers to compensatory changes in synaptic strength or intrinsic excitability that keep neuronal or network activity near a target level when activity is perturbed. It acts in the opposite direction to Hebbian change: sustained overactivity triggers a reduction in excitation, and sustained underactivity triggers an increase. It operates over hours to days, slower than Hebbian LTP and LTD.

## Mechanism
Hebbian rules are positive feedback: strengthening a synapse makes it more likely to be strengthened again, which would drive runaway excitation or silencing without a counterbalance. Homeostatic mechanisms provide negative feedback:

- **Synaptic scaling**: all of a neuron's excitatory synapses are multiplied by a common factor. Turrigiano and colleagues (1998) showed that blocking activity in cortical cultures increased the amplitude of miniature excitatory postsynaptic currents, and raising activity reduced it. Because the change is multiplicative, relative differences among synapses, and thus stored information, are approximately preserved.
- **Intrinsic excitability regulation**: changes in ion-channel expression adjust the neuron's firing threshold or gain.
- **Inhibitory plasticity**: strengthening or weakening inhibition to rebalance excitation and inhibition.
- **Presynaptic homeostasis**: adjusting release probability.

The activity sensor is thought to involve calcium-dependent signaling reporting average firing over long windows, though the molecular details are still being worked out. Homeostatic plasticity has been implicated in the recovery of cortical responses after sensory deprivation, and in sleep-related synaptic downscaling, a proposal known as the synaptic homeostasis hypothesis (Tononi and Cirelli, 2003).

## Role in Active Inference
The connection to the free-energy principle is indirect, and a terminological caution is needed. In the free-energy principle, homeostasis normally refers to an organism keeping its physiological states within viable bounds, described as minimizing surprise about its own states (see [[homeostatic_regulation]]). That is not the same as a neuron holding its firing rate near a set point. A related, more modest link is that homeostatic scaling plays a stabilizing role analogous to a constraint on the complexity or magnitude of parameters, and that regulating average gain resembles setting the overall precision of a population. These parallels have been suggested but are not derived results. What can be said is that any learning scheme with Hebbian-like updates needs some form of normalization to remain bounded, and homeostatic plasticity is the biological candidate for it.

## Related concepts
- [[hebbian_learning]] — positive-feedback rule that homeostasis constrains
- [[synaptic_plasticity]] — the broader family of synaptic change
- [[stability_plasticity]] — the dilemma homeostasis helps resolve
- [[metaplasticity]] — related regulation of the plasticity rule itself
- [[homeostatic_regulation]] — organism-level homeostasis, a different level of description
- [[neural_plasticity]] — parent concept

## References
- Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C., & Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature*.
- Tononi, G., & Cirelli, C. (2003). Sleep and synaptic homeostasis: a hypothesis. *Brain Research Bulletin*.
