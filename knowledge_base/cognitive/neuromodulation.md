---
title: Neuromodulation
description: The regulation of neural circuit function by chemical signals that alter excitability and synaptic strength, the main modulatory systems, and Active Inference proposals linking them to precision.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neurotransmitters
  - precision
  - circuits
semantic_relations:
  - type: relates
    links:
      - "[[dopamine_system]]"
      - "[[cholinergic_system]]"
      - "[[noradrenergic_system]]"
      - "[[serotonin]]"
      - "[[precision_weighting]]"
wikidata: Q905905
---

# Neuromodulation

## Overview
Neuromodulation is the process by which chemical signals change the properties of neurons and synapses so that a circuit's output changes without a change in its anatomical wiring. Whereas fast transmission by glutamate or GABA conveys specific information between connected cells, neuromodulators such as dopamine, noradrenaline, serotonin, acetylcholine and many neuropeptides typically act through G-protein-coupled receptors, over longer times and often through diffuse release (volume transmission). Their effects include changes in intrinsic excitability, synaptic strength, plasticity thresholds and the gain of responses.

## Properties
Marder (2012) emphasises that neuromodulation makes a fixed anatomical circuit multifunctional. Studies of the small crustacean stomatogastric ganglion showed that the same neurons can produce different rhythmic outputs depending on the modulators present. Several features recur across systems: modulators are released by small brainstem or basal forebrain nuclei with widespread projections, their effects depend on receptor subtype and cell type, they act on several timescales, and they are tied to behavioural state, such as arousal, reward, stress, attention and sleep. Pharmacology and disorders illustrate the importance of the systems: levodopa in Parkinson's disease, cholinesterase inhibitors in Alzheimer's disease, and psychedelic and antidepressant drugs acting on serotonin signalling.

## Computational roles
Theories assign modulators to computational variables. Dopamine has been linked to reward prediction error, acetylcholine to expected uncertainty and noradrenaline to unexpected uncertainty (Yu and Dayan, 2005). Other proposals relate modulation to learning rate, exploration or the balance between stability and plasticity (see [[stability_plasticity]]).

## Relation to Active Inference
Active Inference interprets many neuromodulators as encoding precision, the inverse variance of beliefs or prediction errors, which controls how strongly they affect inference. Proposals include acetylcholine for the precision of sensory evidence, with supporting simulations and data on cholinergic manipulation (Moran et al., 2013), dopamine for the precision of beliefs about policies, and noradrenaline and serotonin for other aspects of uncertainty. These mappings are an active research programme with partial experimental support, and rival assignments exist; they should be read as hypotheses, not as a settled functional taxonomy.

## Related concepts
- [[dopamine_system]] — modulator linked to reward and policy precision
- [[cholinergic_system]] — modulator linked to sensory precision and attention
- [[noradrenergic_system]] — modulator linked to arousal and unexpected uncertainty
- [[serotonin]] — modulator with diverse proposed roles
- [[precision_weighting]] — the Active Inference variable modulators are said to set
- [[stability_plasticity]] — the balance that modulation helps regulate

## References
- Yu, A. J., & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. Neuron, 46, 681-692.
- Marder, E. (2012). Neuromodulation of neuronal circuits: back to the future. Neuron, 76, 1-11.
- Moran, R. J., et al. (2013). Free energy, precision and learning: the role of cholinergic neuromodulation. Journal of Neuroscience, 33, 8227-8236.
