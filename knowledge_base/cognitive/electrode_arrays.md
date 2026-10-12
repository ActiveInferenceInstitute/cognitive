---
title: Electrode Arrays
description: Multi-contact devices that record or stimulate neural activity at many sites at once, including Utah arrays, silicon probes and Neuropixels, with their uses in neural coding and brain-machine interfaces.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neurotechnology
  - electrophysiology
  - recording
semantic_relations:
  - type: relates
    links:
      - "[[electrophysiology]]"
      - "[[neural_coding]]"
      - "[[single_unit]]"
      - "[[brain_machine_interfaces]]"
      - "[[population_coding]]"
---

# Electrode Arrays

## Overview
An electrode array is a device carrying many recording or stimulating contacts that is placed in or on neural tissue. By sampling many sites at once, arrays allow simultaneous measurement of populations of neurons, which a single microelectrode cannot provide. They are central to studies of neural coding, and to clinical and research brain-machine interfaces.

## Types
- **Penetrating arrays.** The Utah array is a grid of silicon needles inserted into cortex, typically recording from on the order of a hundred sites, and has been implanted in humans for research on prosthetic control. Tetrodes and linear multi-site silicon probes record from many neurons along a single track.
- **High-density probes.** Neuropixels probes integrate hundreds of recording sites along a thin shank with on-probe amplification and digitisation, allowing activity from large numbers of neurons across brain regions in behaving animals (Jun and colleagues, 2017).
- **Surface arrays.** Electrocorticography grids sit on the cortical surface and record local field potentials with lower spatial resolution but less invasiveness.
- **Stimulating arrays.** Arrays can also deliver current, as in deep brain stimulation or visual and sensory prostheses.

## Signals and limitations
Wide-band recordings contain local field potentials, reflecting synaptic activity in a local volume, and spikes, which are sorted into single units or pooled as multi-unit activity. Spike sorting can be ambiguous with dense or noisy recordings. Tissue response around implants, including glial scarring, can degrade signal quality over months to years, and surgical implantation carries risks, so long-term stability remains an engineering challenge.

## Relation to Active Inference
Electrode arrays are measurement tools. They provide the neural population data against which proposals such as predictive coding and Active Inference can be tested, for example by examining whether neurons encode prediction errors. The tools themselves carry no theoretical commitment.

## Related concepts
- [[electrophysiology]] — the field that uses these devices
- [[neural_coding]] — questions about how neurons represent information
- [[single_unit]] — activity of individual neurons
- [[brain_machine_interfaces]] — a major application
- [[population_coding]] — representation spread across many neurons
- [[spike_trains]] — the discrete events recorded

## References
- Buzsáki, G. (2004). Large-scale recording of neuronal ensembles. Nature Neuroscience, 7, 446-451.
- Hochberg, L. R., Serruya, M. D., Friehs, G. M., Mukand, J. A., Saleh, M., Caplan, A. H., Branner, A., Chen, D., Penn, R. D., & Donoghue, J. P. (2006). Neuronal ensemble control of prosthetic devices by a human with tetraplegia. Nature, 442, 164-171.
- Jun, J. J., et al. (2017). Fully integrated silicon probes for high-density recording of neural activity. Nature, 551, 232-236.
