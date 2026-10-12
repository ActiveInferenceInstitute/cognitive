---
title: Chemogenetics
description: Methods that engineer receptors or channels to respond to otherwise inert small molecules, enabling remote control of defined cell populations, with designer receptors (DREADDs) as the main example.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neuroscience_methods
  - receptor_engineering
  - circuit_manipulation
semantic_relations:
  - type: relates
    links:
      - "[[optogenetics]]"
      - "[[neural_circuits]]"
      - "[[neuromodulation]]"
      - "[[neural_plasticity]]"
      - "[[ion_channels]]"
---

# Chemogenetics

## Overview
Chemogenetics refers to techniques in which a protein is engineered so that it is activated by a synthetic small molecule that has little or no effect on unmodified cells. When the engineered protein is expressed in a chosen cell type, administering the ligand lets researchers raise or lower the activity of that population in a living animal without implanted hardware. The approach is used to test the causal role of circuits in behaviour and physiology.

## Designer receptors
The most widely used class is DREADDs, designer receptors exclusively activated by designer drugs. They were derived from human muscarinic acetylcholine receptors by directed molecular evolution in yeast, selecting variants that lost response to acetylcholine and gained response to clozapine-N-oxide (CNO) (Armbruster et al., 2007).
- **hM3Dq** couples to $G_q$ signalling and typically depolarises and excites neurons.
- **hM4Di** couples to $G_i/G_o$ signalling and typically hyperpolarises and suppresses neurotransmitter release.
- **KORD**, derived from the kappa opioid receptor, responds to a different ligand and allows opposing manipulations in one animal.
Other chemogenetic systems use ligand-gated ion channels engineered from other channels, called pharmacologically selective actuator modules.

## Strengths and caveats
Compared with [[optogenetics]], chemogenetics has coarser time resolution, in the range of tens of minutes to hours, but it needs no light delivery and can influence broad or distributed populations. Specificity depends on the actuator ligand. CNO is converted in the body to clozapine, which itself acts on native receptors, and so controls with ligand-treated animals lacking the receptor are essential (Gomez et al., 2017). More selective agonists have since been introduced. Receptor overexpression and chronic stimulation can also alter cellular physiology.

## Relation to Active Inference
Chemogenetics is an experimental tool, not a theory. Its link to Active Inference lies in testing hypotheses about neuromodulatory control, for example whether manipulating a modulatory population changes behaviour in the way that precision-based accounts predict. Such interpretations depend on the model assumed, and a manipulation that changes behaviour does not by itself identify the computational variable involved.

## Related concepts
- [[optogenetics]] — light-based control with finer timing
- [[neural_circuits]] — the targets of cell-type-specific manipulation
- [[neuromodulation]] — the signalling systems that designer receptors can mimic
- [[neural_plasticity]] — a process that chronic manipulation can probe
- [[ion_channels]] — the proteins engineered in channel-based actuators

## References
- Armbruster, B. N., Li, X., Pausch, M. H., Herlitze, S., & Roth, B. L. (2007). Evolving the lock to fit the key to create a family of G protein-coupled receptors potently activated by an inert ligand. Proceedings of the National Academy of Sciences, 104, 5163-5168.
- Sternson, S. M., & Roth, B. L. (2014). Chemogenetic tools to interrogate brain functions. Annual Review of Neuroscience, 37, 387-407.
- Roth, B. L. (2016). DREADDs for neuroscientists. Neuron, 89, 683-694.
- Gomez, J. L., Bonaventura, J., Lesniak, W., et al. (2017). Chemogenetics revealed: DREADD occupancy and activation via converted clozapine. Science, 357, 503-507.
