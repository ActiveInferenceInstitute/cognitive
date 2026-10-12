---
title: Post-Tetanic Potentiation
description: A short-term increase in synaptic strength lasting seconds to minutes after a burst of high-frequency stimulation, its presynaptic calcium-dependent mechanism, and how it differs from long-term potentiation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - synaptic_plasticity
  - short_term_plasticity
  - synapse
semantic_relations:
  - type: relates
    links:
      - "[[synaptic_plasticity]]"
      - "[[ltp]]"
      - "[[quantal_content]]"
      - "[[calcium_signaling]]"
      - "[[neuromuscular_junction]]"
wikidata: Q7233588
---

# Post-Tetanic Potentiation

## Overview
Post-tetanic potentiation (PTP) is an enhancement of synaptic transmission that follows a tetanus, a train of high-frequency presynaptic stimulation. After the train ends, postsynaptic responses to single presynaptic action potentials remain larger than before for a period of seconds to minutes, and then decay back toward baseline. PTP is one of several forms of short-term synaptic enhancement and was described first at the neuromuscular junction, then at many central synapses.

## Mechanism
The dominant explanation is presynaptic. During the tetanus, calcium enters the terminal with each spike, and the intracellular calcium concentration rises to a level from which it falls only slowly. The elevated calcium, acting through targets that include protein kinase C and the vesicle release machinery, raises the probability of transmitter release and may increase the readily releasable pool of vesicles (Zucker & Regehr, 2002). In quantal terms, PTP raises the mean number of vesicles released per impulse, which is the [[quantal_content]], rather than the postsynaptic response to a single vesicle. Separate processes with different time courses, such as facilitation, augmentation and potentiation, overlap after a tetanus, and their contributions are hard to separate (Fisher et al., 1997).

## Distinction from long-term potentiation
PTP is not the same as [[ltp]]. It decays over a time scale much shorter than the hours to days of long-term potentiation, it does not rely on postsynaptic NMDA receptor activation and it is thought to be expressed presynaptically. The two can occur together after a tetanus, with PTP decaying and leaving any long-term component to be measured. Experimental protocols therefore delay measurement of LTP until PTP has decayed.

## Functional significance
Short-term enhancement alters how synapses filter patterns of activity, and has been proposed to serve as a temporary storage mechanism. Mongillo, Barak and Tsodyks (2008) proposed that calcium-mediated short-term facilitation could hold items in working memory without persistent firing. This idea is influential but contested, and other models rely on persistent activity.

## Relation to Active Inference
The link is an analogy: short-term changes in synaptic efficacy could be read as fast updating of the weights that implement a generative model, but no specific Active Inference model of PTP is established.

## Related concepts
- [[synaptic_plasticity]] — the general category
- [[ltp]] — the long-lasting counterpart
- [[quantal_content]] — the quantity increased in PTP
- [[calcium_signaling]] — the trigger
- [[receptor_desensitization]] — a process that reduces rather than enhances transmission

## References
- Zucker, R. S., & Regehr, W. G. (2002). Short-term synaptic plasticity. Annual Review of Physiology, 64, 355-405.
- Fisher, S. A., Fischer, T. M., & Carew, T. J. (1997). Multiple overlapping processes underlying short-term synaptic enhancement. Trends in Neurosciences, 20, 170-177.
- Mongillo, G., Barak, O., & Tsodyks, M. (2008). Synaptic theory of working memory. Science, 319, 1543-1546.
