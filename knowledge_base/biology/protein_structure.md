---
title: Protein Structure
description: The hierarchical organisation of proteins from amino acid sequence to folded three-dimensional assemblies, how structure is determined and predicted, and the thermodynamic principle behind folding.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - proteins
  - structural_biology
  - molecular_biology
semantic_relations:
  - type: relates
    links:
      - "[[protein_folding]]"
      - "[[protein_synthesis]]"
      - "[[enzyme_kinetics]]"
      - "[[molecular_biology]]"
      - "[[cell_biology]]"
wikidata: Q735188
---

# Protein Structure

## Overview
A protein is a chain of amino acids that folds into a specific three-dimensional shape, and that shape determines what it does, whether as an enzyme, a channel, a structural element or a signalling molecule. Structure is described at four conventional levels.

## Levels of organisation
- **Primary structure** is the linear sequence of amino acids, encoded by a gene.
- **Secondary structure** consists of local, regular patterns held by hydrogen bonds in the backbone, chiefly the alpha helix (Pauling et al., 1951) and the beta sheet.
- **Tertiary structure** is the overall fold of a single chain, stabilised by hydrophobic packing, hydrogen bonds, salt bridges and, in some proteins, disulfide bonds.
- **Quaternary structure** is the arrangement of several chains into a complex, as in haemoglobin.

Not every protein folds into a single stable structure. Many have intrinsically disordered regions that adopt shape only on binding a partner, and these regions are often important in signalling and regulation.

## Folding and determination
Anfinsen (1973) showed through refolding experiments that the native structure of a small protein forms spontaneously under suitable conditions and is determined by its sequence. This is the thermodynamic hypothesis: the native state corresponds to the minimum of Gibbs free energy of the system. Folding is not a random search, since a chain cannot sample all conformations in a feasible time, and it is usually described as descent on a funnel-shaped energy landscape (see [[protein_folding]]). In cells, chaperone proteins assist folding and prevent aggregation.

Structures are determined experimentally by X-ray crystallography, nuclear magnetic resonance spectroscopy and cryo-electron microscopy. Computational prediction advanced sharply with the deep-learning system AlphaFold 2, which predicted many structures with accuracy close to experimental in blind assessment (Jumper et al., 2021).

## Relation to Active Inference
The connection is limited and should not be confused with a shared quantity. The free energy that governs folding is the physical Gibbs or Helmholtz free energy, whereas the variational free energy of Active Inference is an information-theoretic bound on surprise. Both involve minimisation over a landscape, which makes the relation an analogy, though their formal connections are studied in non-equilibrium thermodynamics (see [[free_energy_landscape]]).

## Related concepts
- [[protein_folding]] — the process by which a chain reaches its structure
- [[protein_synthesis]] — the production of the amino acid chain
- [[enzyme_kinetics]] — how structure underlies catalytic activity
- [[free_energy_landscape]] — landscape picture used for folding and beyond
- [[cytoskeleton]] — a cell system built from protein polymers

## References
- Anfinsen, C. B. (1973). Principles that govern the folding of protein chains. Science, 181, 223-230.
- Pauling, L., Corey, R. B., & Branson, H. R. (1951). The structure of proteins: two hydrogen-bonded helical configurations of the polypeptide chain. Proceedings of the National Academy of Sciences USA, 37, 205-211.
- Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature, 596, 583-589.
