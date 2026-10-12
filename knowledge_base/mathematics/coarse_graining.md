---
title: Coarse-Graining
description: The replacement of a detailed microscopic description by a reduced macroscopic one through grouping or averaging of states or variables, its formal properties for probabilities and dynamics, and its role in the renormalization group and in multiscale accounts of inference.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - statistical_physics
  - renormalization
  - information_theory
  - multiscale
aliases:
  - coarse-graining
  - coarse graining
  - coarse-grained description
semantic_relations:
  - type: relates
    links:
      - "[[renormalization_group]]"
      - "[[exact_renormalization_group]]"
      - "[[causal_emergence]]"
      - "[[state_space_abstraction]]"
      - "[[scale_invariance]]"
      - "[[information_theory]]"
---

# Coarse-Graining

## Overview
Coarse-graining is the construction of a reduced description of a system by discarding detail. Microscopic variables are grouped or averaged, and the resulting macroscopic variables are used to state laws or predictions at a larger scale. A thermodynamic temperature summarises the motions of enormous numbers of molecules, and a block of spins on a lattice can be summarised by a single spin. Coarse-graining is many-to-one, so information about the microstate is lost, and which detail is safe to lose depends on the question being asked.

## Formal description
Let $\pi:\mathcal{X}\to\mathcal{Y}$ map each micro-state $x$ to a macro-state $y=\pi(x)$. A probability distribution $p$ over $\mathcal{X}$ induces a macro distribution by summing over the group,
$$q(y)=\sum_{x:\,\pi(x)=y}p(x).$$
Two general facts follow from the data-processing properties of information measures. The entropy of the macro variable cannot exceed that of the micro variable, $H(Y)\le H(X)$, and the Kullback-Leibler divergence between two distributions cannot increase under coarse-graining, $D_{\mathrm{KL}}(q\,\|\,q')\le D_{\mathrm{KL}}(p\,\|\,p')$. For dynamics the situation is less forgiving: a coarse-grained version of a Markov process is again Markov only for special groupings, called lumpable partitions, so macro-level laws are in general approximate.

## Block-spin coarse-graining
Kadanoff (1966) proposed to treat a lattice of spins near a critical point by grouping neighbouring spins into blocks of linear size $b$, assigning each block a single effective spin, and rescaling lengths by $b$. The effective couplings of the blocked system are functions of the original ones. Iterating the map defines a flow in the space of couplings. Wilson (1971) made this picture into the renormalization group (see [[renormalization_group]]), where fixed points of the flow describe scale-invariant behaviour. The continuous-scale versions of the flow are collected under [[exact_renormalization_group]].

## Use in statistics and learning
Mehta and Schwab (2014) showed an exact mapping between one variational form of Kadanoff's block-spin scheme and the layered structure of deep networks built from restricted Boltzmann machines, which suggests that successive layers can be read as successive coarse-grainings (see also [[bayesian_renormalization]]). Hoel, Albantakis and Tononi (2013) used coarse-graining of a system's transition probabilities to ask whether a macro description can carry more causal information than the micro one, a claim still debated (see [[causal_emergence]]).

## Relation to Active Inference
The connection is interpretive. Hierarchical generative models in Active Inference are often described as having higher levels that represent coarser summaries of the levels below, in space or in time, which is a form of coarse-graining (see [[state_space_abstraction]]). The nested Markov blankets used in the free-energy principle describe a similar separation of scales. None of this implies that a given hierarchical model is the exact output of a renormalization procedure, and the choice of grouping remains a modelling decision.

## References
- Kadanoff, L. P. (1966). Scaling laws for Ising models near $T_c$. Physics Physique Fizika, 2(6), 263-272. doi:10.1103/physicsphysiquefizika.2.263
- Wilson, K. G. (1971). Renormalization group and critical phenomena. I. Renormalization group and the Kadanoff scaling picture. Physical Review B, 4(9), 3174-3183. doi:10.1103/physrevb.4.3174
- Mehta, P., & Schwab, D. J. (2014). An exact mapping between the variational renormalization group and deep learning. arXiv:1410.3831.
- Hoel, E. P., Albantakis, L., & Tononi, G. (2013). Quantifying causal emergence shows that macro can beat micro. Proceedings of the National Academy of Sciences, 110(49), 19790-19795. doi:10.1073/pnas.1314922110
- Goldenfeld, N. (1992). Lectures on Phase Transitions and the Renormalization Group. Addison-Wesley.
