---
title: High-Density Probes
description: Multi-site silicon electrode probes such as Neuropixels that record from hundreds of neurons simultaneously, their design, applications, practical challenges and role in testing population-level theories.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - electrophysiology
  - recording_methods
  - neural_coding
semantic_relations:
  - type: relates
    links:
      - "[[electrophysiology]]"
      - "[[spike_sorting]]"
      - "[[population_coding]]"
      - "[[neural_coding]]"
      - "[[eeg_recording]]"
      - "[[brain_machine_interfaces]]"
---

# High-Density Probes

## Overview
High-density probes are implantable electrodes with many closely spaced recording sites on a thin shaft, built with semiconductor fabrication methods. They record extracellular voltage from a column of tissue and so capture the spiking of many neurons at once, often across several brain regions along the shaft. They replaced earlier practice in which one or a few wire electrodes were advanced by hand while single neurons were isolated one at a time (Buzsaki, 2004).

## Design and capabilities
The Neuropixels probe is the most widely used example. A single shank about a centimetre long carries 960 recording sites, of which 384 channels can be recorded at once, with the amplifiers and digitisation built into the probe base (Jun et al., 2017). A later version has multiple shanks and is small and light enough for chronic implantation and recording in freely moving animals (Steinmetz et al., 2021). Probes of this type have also been used for intraoperative recordings in human cortex. Other approaches include planar arrays with a grid of shanks, flexible polymer electrodes and optical methods that image calcium signals, each with different trade-offs between the number of cells, temporal resolution and invasiveness.

## Uses and challenges
- **Population analyses.** Simultaneous recording lets researchers study correlations, low-dimensional dynamics and coding across populations on single trials, which are inaccessible with sequential recordings.
- **Multi-region studies.** Probes spanning cortex, thalamus and midbrain reveal how activity relates across structures during behaviour.
- **Data handling.** Output volumes are large, and assigning spikes to neurons requires automated spike sorting (see [[spike_sorting]]) that carries its own error rates and need for curation.
- **Stability and tissue effects.** Tissue movement relative to the probe causes drift, and implants may cause local damage or an inflammatory response over long periods.

## Relation to Active Inference
The relation is methodological. Proposals in predictive coding and Active Inference concern coding of expectations and prediction errors in identified cell classes and layers, and testing them requires simultaneous recordings across layers and areas. High-density probes make some of these tests possible, but the technique is neutral between theories, and what the data show depends on analysis choices.

## Related concepts
- [[electrophysiology]] — the family of methods for recording neural electrical activity
- [[spike_sorting]] — the step that turns raw voltage into single-neuron spike trains
- [[population_coding]] — information carried by the joint activity of many cells
- [[neural_coding]] — the general question these recordings address
- [[eeg_recording]] — a non-invasive method with much coarser spatial resolution

## References
- Jun, J. J., Steinmetz, N. A., Siegle, J. H., Denman, D. J., Bauza, M., Barbarits, B., et al. (2017). Fully integrated silicon probes for high-density recording of neural activity. Nature, 551, 232-236.
- Steinmetz, N. A., Aydin, C., Lebedeva, A., Okun, M., Pachitariu, M., Bauza, M., et al. (2021). Neuropixels 2.0: A miniaturized high-density probe for stable, long-term brain recordings. Science, 372, eabf4588.
- Buzsaki, G. (2004). Large-scale recording of neuronal ensembles. Nature Neuroscience, 7, 446-451.
