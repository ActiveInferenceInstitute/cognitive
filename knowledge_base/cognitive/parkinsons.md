---
title: Parkinson's Disease
description: A progressive neurodegenerative movement disorder caused by loss of nigrostriatal dopamine, covering its motor and non-motor features, basal ganglia pathophysiology, and Active Inference accounts of dopamine and action.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neurodegeneration
  - basal_ganglia
  - dopamine
  - movement_disorder
semantic_relations:
  - type: relates
    links:
      - "[[motor_control]]"
      - "[[dopamine_system]]"
      - "[[motor_learning]]"
      - "[[ventral_striatum]]"
      - "[[precision_weighting]]"
---

# Parkinson's Disease

## Overview
Parkinson's disease is a progressive neurodegenerative disorder defined clinically by bradykinesia (slowness and reduced amplitude of movement) together with rest tremor, rigidity, or both. It was first described by James Parkinson in 1817. The characteristic pathology is loss of dopaminergic neurons in the substantia nigra pars compacta, accompanied by intraneuronal alpha-synuclein aggregates known as Lewy bodies. Motor signs usually appear only after a large fraction of nigrostriatal dopamine has been lost.

## Pathophysiology and features
- **Basal ganglia imbalance.** The classical rate model holds that dopamine depletion in the striatum weakens the direct pathway and strengthens the indirect pathway, increasing inhibitory output from the internal globus pallidus onto the thalamus and reducing drive to motor cortex (Albin et al., 1989). The model is a useful simplification; abnormal oscillatory synchrony in the beta band is also implicated.
- **Non-motor features.** Sleep disturbance, loss of smell, constipation, depression, anxiety and cognitive decline are common, and some can precede motor onset.
- **Treatment.** Dopamine replacement with levodopa or dopamine agonists is the mainstay, and deep brain stimulation of the subthalamic nucleus or globus pallidus is used in selected patients. Treatment relieves symptoms but does not halt degeneration.

## Relation to Active Inference
Dopamine has two prominent computational interpretations: signalling reward prediction error in reinforcement learning, and, in Active Inference, encoding the precision of beliefs about which policy to pursue. On the second reading, depleted dopamine lowers the confidence with which one action sequence is preferred over others, so that the agent fails to commit to movement. Simulations by Friston and colleagues (2012) used this idea to model dopaminergic lesions as reduced precision of prediction errors, producing behaviours the authors described as reminiscent of those seen in Parkinson's disease. This is a theoretical proposal supported by modelling and an active research programme, not an established mechanism. It is also compatible with accounts in which the underlying difficulty lies in automatic, habitual action selection (see [[procedural_memory]]).

## Related concepts
- [[motor_control]] — the function most visibly disrupted
- [[dopamine_system]] — the neuromodulatory system lost in the disease
- [[motor_learning]] — skill learning and automatic movement are affected
- [[ventral_striatum]] — striatal region involved in reward and motivation
- [[precision_weighting]] — the Active Inference reading of dopamine
- [[procedural_memory]] — habit and sequence knowledge dependent on the striatum

## References
- Parkinson, J. (1817). An Essay on the Shaking Palsy. London: Sherwood, Neely, and Jones.
- Albin, R. L., Young, A. B., & Penney, J. B. (1989). The functional anatomy of basal ganglia disorders. Trends in Neurosciences, 12, 366-375.
- Friston, K. J., et al. (2012). Dopamine, affordance and active inference. PLoS Computational Biology, 8, e1002327.
