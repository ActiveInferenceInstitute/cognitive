---
title: Binding Sites
description: The regions of proteins, nucleic acids and other macromolecules where ligands attach through complementary shape and non-covalent interactions, with affinity, occupancy and models of specificity.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - molecular_biology
  - ligand_binding
  - biochemistry
semantic_relations:
  - type: relates
    links:
      - "[[allosteric_effects]]"
      - "[[enzyme_kinetics]]"
      - "[[transcription_factors]]"
      - "[[protein_structure]]"
      - "[[receptor_phosphorylation]]"
---

# Binding Sites

## Overview
A binding site is the region of a macromolecule that a ligand attaches to. The ligand may be a substrate, a signalling molecule, an ion, a drug or another macromolecule, and the site may be on a protein, as in an enzyme active site or a receptor pocket, or on DNA or RNA, as in the sequence recognised by a transcription factor. Binding is selective because the site matches the ligand in shape and in distribution of charge and polarity.

## Mechanism and quantities
Binding is mediated by non-covalent interactions: hydrogen bonds, ionic interactions, van der Waals contacts and the hydrophobic effect. Each is weak, but many together give specific, reversible binding. For a single site, $P+L\rightleftharpoons PL$ with dissociation constant
$$K_d=\frac{[P][L]}{[PL]},$$
and the fraction of sites occupied at equilibrium is $\theta=[L]/(K_d+[L])$. A smaller $K_d$ means higher affinity, and $\theta=1/2$ when $[L]=K_d$. Rate constants determine how quickly equilibrium is reached, and binding kinetics can matter as much as affinity for biological function.

## Models of recognition
Fischer's lock-and-key picture (1894) treats the site as rigidly complementary to its ligand. Koshland's induced-fit model (1958) allows the site to adjust its shape on binding. The conformational-selection view holds that the protein samples several conformations and the ligand binds and stabilises one of them. Experimental evidence supports different mechanisms for different systems, and the two latter views are not mutually exclusive.

## Types of site
- **Orthosteric sites** bind the endogenous ligand, such as the substrate pocket of an enzyme.
- **Allosteric sites** are distinct locations that modulate the orthosteric site (see [[allosteric_effects]]).
- **Regulatory modification sites**, such as residues phosphorylated in signalling (see [[receptor_phosphorylation]]).
- **Nucleic acid sites**, short sequence motifs recognised by [[transcription_factors]].

## Relation to Active Inference
Binding sites are the molecular interface at which a cell registers its environment, and receptor occupancy is the first stage of sensing. In Active Inference accounts of cells as inferential systems, this interface corresponds loosely to the sensory part of a Markov blanket. The correspondence is an interpretive framing and is not a molecular theory. Physical limits on how well a cell can infer a concentration from receptor events are studied in gradient-sensing theory.

## Related concepts
- [[allosteric_effects]] — regulation through a second site
- [[enzyme_kinetics]] — how substrate binding translates into reaction rates
- [[transcription_factors]] — proteins that bind specific DNA sites
- [[protein_structure]] — the three-dimensional basis of site geometry
- [[receptor_phosphorylation]] — modification that alters receptor binding and signalling

## References
- Fischer, E. (1894). Einfluss der Configuration auf die Wirkung der Enzyme. Berichte der Deutschen Chemischen Gesellschaft, 27, 2985-2993.
- Koshland, D. E. (1958). Application of a theory of enzyme specificity to protein synthesis. Proceedings of the National Academy of Sciences, 44, 98-104.
- Berg, J. M., Tymoczko, J. L., Gatto, G. J., & Stryer, L. (2019). Biochemistry (9th ed.). W. H. Freeman.
