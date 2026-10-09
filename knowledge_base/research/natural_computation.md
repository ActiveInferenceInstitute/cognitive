---
title: Natural Computation
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - research
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/research/biological_intelligence]]"
      - "[[knowledge_base/biology/systems_biology]]"
      - "[[knowledge_base/mathematics/thermodynamics]]"
      - "[[knowledge_base/mathematics/non_equilibrium_thermodynamics]]"
      - "[[knowledge_base/mathematics/information_theory]]"
      - "[[knowledge_base/biology/gene_regulatory_networks]]"
      - "[[knowledge_base/biology/cellular_intelligence]]"
      - "[[knowledge_base/biology/metabolic_networks]]"
      - "[[knowledge_base/cognitive/emergence_self_organization]]"
      - "[[knowledge_base/research/bio_inspired_computing]]"
---

# Natural Computation

## Overview

Natural computation is the study of computation carried out by natural systems, and of computing methods inspired by them. The two directions are distinct. One asks what biological, chemical and physical processes compute and how efficiently. The other asks what computing can learn from them. Examples include cellular automata, molecular and DNA computing, gene regulatory networks, and the information processing performed by single cells.

## Main topics

- **Cellular automata.** Von Neumann's self-reproducing automaton, Conway's Game of Life, and Wolfram's classification of one-dimensional rules show that simple local rules can produce complex global behaviour and, for some rules, universal computation.
- **Molecular computing.** Adleman (1994) solved a small instance of a combinatorial problem using DNA strands, launching DNA computing. It is a proof of principle, and DNA computers have not competed with electronic ones on speed. Later work built logic circuits and storage from DNA and RNA.
- **Cellular information processing.** Bacteria and eukaryotic cells sense and respond to chemical gradients with noisy receptors. Berg and Purcell (1977) derived physical limits on the precision of concentration sensing, and experimental work has asked how close cells come to them.
- **Gene regulatory and metabolic networks** as computing circuits, with attractors that correspond to cell types.
- **Physical limits.** Landauer (1961) showed that erasing a bit has a minimum thermodynamic cost, setting a bound that real devices, biological and electronic, operate far above.

## Connection to Active Inference

The connection runs through thermodynamics and is partly formal, partly analogical. The free energy principle uses a variational free energy that is an information-theoretic bound, and the Landauer-type results relate information processing to thermodynamic free energy. The two quantities are formally related in specific stochastic-thermodynamics settings, but they are not identical, and the general bridge between them is under active research, not settled. Work on the energy efficiency of neural information, such as Sengupta, Stemmler and Friston (2013), links the variational formulation to metabolic cost.

For cellular computation, the proposal is that chemotaxis and similar behaviours can be described as inference about the environment under a simple generative model. This agrees with the physics-of-sensing literature in that both treat the cell as an estimator, but it has not been shown to predict anything beyond that literature. Cellular automata, on the other hand, are better seen as model systems for emergence than as examples of Active Inference.

## In this repository

See [[knowledge_base/mathematics/thermodynamics]], [[knowledge_base/mathematics/non_equilibrium_thermodynamics]], [[knowledge_base/mathematics/information_theory]], [[knowledge_base/biology/gene_regulatory_networks]], [[knowledge_base/biology/cellular_intelligence]], [[knowledge_base/biology/metabolic_networks]] and [[knowledge_base/cognitive/emergence_self_organization]]. For the bio-inspired engineering side see [[knowledge_base/research/bio_inspired_computing]].

## References

- Landauer, R. (1961). Irreversibility and heat generation in the computing process. *IBM Journal of Research and Development*, 5, 183-191.
- Berg, H. C., & Purcell, E. M. (1977). Physics of chemoreception. *Biophysical Journal*, 20, 193-219.
- Adleman, L. M. (1994). Molecular computation of solutions to combinatorial problems. *Science*, 266, 1021-1024.
- Sengupta, B., Stemmler, M. B., & Friston, K. J. (2013). Information and efficiency in the nervous system: a synthesis. *PLoS Computational Biology*, 9, e1003157.

## Related Resources

- [[knowledge_base/research/biological_intelligence]]
- [[knowledge_base/biology/systems_biology]]
