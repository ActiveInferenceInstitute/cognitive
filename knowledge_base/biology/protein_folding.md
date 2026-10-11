---
title: Protein Folding
description: The process by which a polypeptide chain acquires its functional three-dimensional structure, covering Anfinsen's hypothesis, energy landscape theory, chaperones, misfolding, and computational structure prediction.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - structural_biology
  - biophysics
  - energy_landscape
semantic_relations:
  - type: relates
    links:
      - "[[protein_synthesis]]"
      - "[[structural_proteins]]"
      - "[[free_energy_landscape]]"
      - "[[thermodynamics]]"
      - "[[molecular_biology]]"
---

# Protein Folding

## Overview
After a ribosome synthesises a chain of amino acids (see [[protein_synthesis]]), most proteins fold into a specific compact three-dimensional structure that determines their function. Folding typically occurs on timescales from microseconds to seconds, although it can take much longer for large or multi-domain proteins. The protein folding problem has three related parts: predicting a native structure from sequence, explaining the folding pathway, and understanding why the process is so efficient.

## Thermodynamic basis
In classic experiments on ribonuclease, Anfinsen showed that a denatured protein could refold spontaneously in vitro, supporting the hypothesis that the native structure is determined by the amino acid sequence and corresponds to a minimum of Gibbs free energy under physiological conditions. Principal contributions to stability include the hydrophobic effect, which buries nonpolar side chains, hydrogen bonding, electrostatic interactions and van der Waals packing, opposed by the loss of conformational entropy of the chain.

Levinthal's paradox points out that a chain cannot find its native state by random search through all conformations. The resolution offered by energy landscape theory is that the landscape is funnel-shaped and minimally frustrated, so that partial structure formation lowers energy and guides the chain toward the native state through many possible routes (see [[free_energy_landscape]]).

## In the cell
- **Chaperones** assist folding of proteins that would otherwise aggregate, particularly in the crowded cellular environment, but they do not specify the final structure.
- **Misfolding** and aggregation are linked to disease, including amyloid disorders and prion diseases.
- **Intrinsically disordered proteins** lack a single stable fold yet remain functional, showing that structure is not always required.
- **Computational prediction.** Deep learning systems, notably AlphaFold, now predict structures of many single-chain proteins with high accuracy from sequence, though they do not by themselves explain kinetic pathways.

## Relation to Active Inference
The overlap is largely one of vocabulary and mathematics. Both folding theory and Active Inference describe systems descending a free energy landscape, and both use statistical mechanics. The thermodynamic free energy of a chain and the variational free energy of a belief distribution are different quantities, so this is an analogy and not an equivalence. See [[free_energy_principle]] for the variational quantity.

## Related concepts
- [[protein_synthesis]] — production of the chain that then folds
- [[structural_proteins]] — proteins whose folded form provides mechanical support
- [[free_energy_landscape]] — the landscape picture of conformational search
- [[thermodynamics]] — the energetic principles governing stability
- [[molecular_biology]] — the wider field in which structure and function are studied

## References
- Anfinsen, C. B. (1973). Principles that govern the folding of protein chains. Science, 181, 223-230.
- Bryngelson, J. D., Onuchic, J. N., Socci, N. D., & Wolynes, P. G. (1995). Funnels, pathways, and the energy landscape of protein folding: a synthesis. Proteins, 21, 167-195.
- Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature, 596, 583-589.
