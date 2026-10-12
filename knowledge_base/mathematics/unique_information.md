---
title: Unique Information
description: The information a source variable provides about a target that no other source provides, defined within partial information decomposition and by the Bertschinger et al. optimisation over marginal-preserving distributions.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - information_decomposition
  - multivariate_dependence
semantic_relations:
  - type: relates
    links:
      - "[[information_decomposition]]"
      - "[[synergistic_information]]"
      - "[[redundant_information]]"
      - "[[conditional_mutual_information]]"
      - "[[mutual_information]]"
---

# Unique Information

## Overview
Unique information is the part of the information that one source variable carries about a target which is not available from another source. In partial information decomposition (PID), the information that two sources $X$ and $Y$ provide about a target $T$ is split into redundant, unique and synergistic parts. The unique part of $X$, written $\mathrm{Unq}(T;X\setminus Y)$, answers the question of what $X$ tells about $T$ that $Y$ does not.

## Relations among the terms
In the two-source case,
$$I(T;X)=\mathrm{Red}+\mathrm{Unq}_X,\qquad I(T;X\mid Y)=\mathrm{Unq}_X+\mathrm{Syn}.$$
Hence if redundancy is defined, the unique parts follow, and conversely. Unique information is non-negative by construction and is zero for a source whose information about $T$ is entirely contained in the other. Two simple cases: if $T=X$ and $Y$ is independent of $X$, all of $I(T;X)$ is unique to $X$; if $X=Y=T$, the information is entirely redundant and neither source has a unique part.

## The Bertschinger definition
Bertschinger, Rauh, Olbrich, Jost and Ay (2014) proposed to define unique information through the set of distributions that agree with the observed pairwise marginals of target and each source,
$$\Delta_P=\{Q:\;Q(X,T)=P(X,T),\;Q(Y,T)=P(Y,T)\}.$$
The unique information is the smallest conditional mutual information attainable in that set,
$$\mathrm{Unq}(T;X\setminus Y)=\min_{Q\in\Delta_P}I_Q(T;X\mid Y).$$
The reasoning is that unique information should depend only on the source-target marginals, since how $X$ and $Y$ relate to each other cannot change what each tells about $T$. Redundancy and synergy are then obtained from the identities above. Other definitions exist, they disagree on some examples, and no single choice is universally accepted.

## Relation to Active Inference
The relevance is indirect. In a system with a Markov blanket, one may ask which of several sensory or internal variables carry information about an external state that the others lack, and PID provides a vocabulary for such questions. The free energy principle itself does not require or derive this decomposition, and applications are analytic tools in the study of neural and agent models.

## Related concepts
- [[information_decomposition]] — the framework that defines unique, redundant and synergistic parts
- [[synergistic_information]] — information available only jointly
- [[redundant_information]] — information that both sources carry
- [[conditional_mutual_information]] — the quantity minimised in the definition
- [[mutual_information]] — the total shared with the target

## References
- Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014). Quantifying unique information. Entropy, 16, 2161-2183.
- Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information. arXiv:1004.2515.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
