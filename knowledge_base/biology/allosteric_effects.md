---
title: Allosteric Effects
description: Changes in the activity or ligand affinity of a protein caused by binding of an effector at a site distinct from the active site, including the concerted and sequential models of cooperativity.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - molecular_biology
  - protein_regulation
  - cooperativity
semantic_relations:
  - type: relates
    links:
      - "[[binding_sites]]"
      - "[[enzyme_kinetics]]"
      - "[[protein_structure]]"
      - "[[receptor_desensitization]]"
      - "[[neuromodulation]]"
---

# Allosteric Effects

## Overview
An allosteric effect occurs when binding of a molecule at one site of a protein changes the protein's behaviour at another site. The effector may be a substrate binding at another subunit (homotropic) or a different molecule (heterotropic), and it may increase or decrease activity. Allostery lets proteins act as switches and sensors, and it underlies feedback regulation of metabolism, oxygen transport by haemoglobin, and the action of many drugs that modulate receptors without competing with the natural ligand.

## Models
The effect is explained by a change in conformational state.
- **Concerted (Monod-Wyman-Changeux) model.** A multimeric protein exists in two states, tense $T$ and relaxed $R$, with all subunits in the same state. In the absence of ligand, $L=[T]/[R]$ and the ligand binds the $R$ state with dissociation constant $K_R$ and the $T$ state with $K_T$, with $c=K_R/K_T<1$. For $n$ identical sites the fractional saturation at reduced ligand concentration $\alpha=[S]/K_R$ is
$$Y=\frac{\alpha(1+\alpha)^{n-1}+Lc\,\alpha(1+c\alpha)^{n-1}}{(1+\alpha)^n+L(1+c\alpha)^n}.$$
Binding shifts the equilibrium toward $R$, producing a sigmoidal curve.
- **Sequential (Koshland-Némethy-Filmer) model.** Binding to one subunit induces a conformational change in that subunit that alters its neighbours' affinity, so intermediate hybrid states exist.

Real proteins often lie between these limits. Cooperativity is commonly summarised by the Hill coefficient, whose value is at most the number of binding sites for positive cooperativity.

## Examples and extensions
Haemoglobin shows cooperative oxygen binding, modulated by protons, carbon dioxide and 2,3-bisphosphoglycerate. Phosphofructokinase and aspartate transcarbamoylase are classic regulated enzymes. Modern views extend allostery to proteins without a large conformational change, where changes in dynamics or entropy transmit the signal, and to the idea that every protein is potentially allosteric.

## Relation to Active Inference
The connection is an analogy. A signal at one location that tunes the gain of a response at another resembles the modulation of sensitivity attributed to precision weighting, and receptors whose responses are tuned by allosteric modulators, such as benzodiazepines at GABA-A receptors, are a concrete biological case (see [[neuromodulation]]). Active Inference does not model protein conformations, and this correspondence concerns function, not mechanism. In cellular biology, allosteric feedback is one of the means by which a cell regulates its internal state.

## Related concepts
- [[binding_sites]] — the orthosteric and allosteric locations where ligands attach
- [[enzyme_kinetics]] — rate laws that cooperativity changes from hyperbolic to sigmoidal
- [[protein_structure]] — the conformational states on which the models depend
- [[receptor_desensitization]] — a regulatory change in receptor responsiveness
- [[neuromodulation]] — gain control by modulators acting on receptors

## References
- Monod, J., Wyman, J., & Changeux, J.-P. (1965). On the nature of allosteric transitions: A plausible model. Journal of Molecular Biology, 12, 88-118.
- Koshland, D. E., Némethy, G., & Filmer, D. (1966). Comparison of experimental binding data and theoretical models in proteins containing subunits. Biochemistry, 5, 365-385.
- Perutz, M. F. (1970). Stereochemistry of cooperative effects in haemoglobin. Nature, 228, 726-734.
- Nelson, D. L., & Cox, M. M. (2021). Lehninger Principles of Biochemistry (8th ed.). W. H. Freeman.
