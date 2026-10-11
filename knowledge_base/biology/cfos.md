---
title: c-Fos
description: An immediate early gene and its protein product, induced rapidly by neuronal activity and used as a marker of recently active cells, and the limits of that use in neuroscience.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - immediate_early_genes
  - activity_markers
  - transcription_factors
semantic_relations:
  - type: relates
    links:
      - "[[bdnf]]"
      - "[[gene_expression]]"
      - "[[transcription_factors]]"
      - "[[memory_formation]]"
      - "[[synaptic_plasticity]]"
---

# c-Fos

## Overview
c-Fos is the protein encoded by the gene *Fos*, a member of the immediate early genes. These genes are switched on within minutes of a cellular stimulus without needing new synthesis of other proteins. In neurons, c-Fos expression increases after strong or sustained activity, and this has made it one of the most widely used markers of which cells were recently engaged by a stimulus or behaviour.

## Mechanism
c-Fos is a basic leucine-zipper transcription factor. It forms dimers with members of the Jun family to make the AP-1 complex, which binds DNA and regulates target genes (see [[transcription_factors]]). In neurons, depolarisation and calcium entry activate signalling cascades, including kinases that act on transcription factors such as CREB and SRF, which drive transcription of *Fos*. The mRNA peaks within roughly half an hour and protein accumulates over one to two hours before falling, so expression reports activity over a window of about an hour or more and not in real time.

## Use as an activity marker
Morgan et al. (1987) showed that seizures induce widespread c-fos expression in the rodent brain, and Sagar, Sharp and Curran (1988) proposed mapping the protein as a cellular-resolution complement to metabolic mapping. Researchers now count Fos-positive cells after behavioural tasks to identify regions that responded. Genetic tools later used the *Fos* promoter to drive permanent labels or optogenetic actuators in the cells active during an experience, which allowed manipulation of putative memory traces, or engrams (Liu et al., 2012).

## Limitations
- Not all active neurons express c-Fos, and expression differs between cell types and regions.
- Reduced activity and inhibition are not detected.
- Its slow time course cannot resolve events within seconds.
- Expression can reflect stress, novelty or handling, so controls are required.

## Relation to Active Inference
There is no direct theoretical link. The marker is an experimental tool, though activity-dependent gene expression is part of the molecular route from neural activity to the plasticity that models describe as learning (see [[synaptic_plasticity]]).

## Related concepts
- [[bdnf]] — another gene induced by activity that supports plasticity
- [[gene_expression]] — the process by which genes are read out
- [[transcription_factors]] — proteins that regulate transcription
- [[memory_formation]] — a process for which Fos-tagged ensembles are studied
- [[fear_conditioning]] — a paradigm commonly used with Fos mapping

## References
- Morgan, J. I., Cohen, D. R., Hempstead, J. L., & Curran, T. (1987). Mapping patterns of c-fos expression in the central nervous system after seizure. Science, 237, 192-197.
- Sagar, S. M., Sharp, F. R., & Curran, T. (1988). Expression of c-fos protein in brain: metabolic mapping at the cellular level. Science, 240, 1328-1331.
- Liu, X., et al. (2012). Optogenetic stimulation of a hippocampal engram activates fear memory recall. Nature, 484, 381-385.
