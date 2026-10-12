---
title: Redundant Information
description: Information about a target that several sources carry in common, its role in partial information decomposition, proposed measures, and the older sense of redundancy in coding and sensory processing.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - partial_information_decomposition
  - redundancy
semantic_relations:
  - type: relates
    links:
      - "[[information_decomposition]]"
      - "[[synergistic_information]]"
      - "[[unique_information]]"
      - "[[mutual_information]]"
      - "[[information_theory]]"
---

# Redundant Information

## Overview
Redundant information is information about a target $T$ that is available separately from each of several source variables $X_1,\dots,X_k$. If two sensors both report the same feature of a stimulus, that feature is redundantly encoded. Redundancy is one of the components in [[information_decomposition]], where $I(T;X_1,X_2)=\mathrm{Red}+\mathrm{Unq}_1+\mathrm{Unq}_2+\mathrm{Syn}$. It is the part that survives if either source is removed.

## Requirements and measures
Williams and Beer (2010) proposed that any measure of redundancy $I_\cap(T;X_1,\dots,X_k)$ should satisfy symmetry in the sources, self-redundancy ($I_\cap(T;X)=I(T;X)$) and monotonicity when sources are added. These imply that redundancy cannot exceed the information in any single source: $\mathrm{Red}\le\min_i I(T;X_i)$. Their measure $I_{\min}$ sums over target values $t$ the smallest specific information any source supplies,
$$I_{\min}(T;X_1,X_2)=\sum_t p(t)\,\min_{i}\,I(T{=}t;X_i),\qquad I(T{=}t;X)=\sum_x p(x\mid t)\ln\frac{p(t\mid x)}{p(t)} .$$
Critics note that $I_{\min}$ can report redundancy when two sources carry equal amounts of information about different aspects of the target. Alternatives measure redundancy through shared structure, such as the bivariate measure of Harder et al. (2013), or define unique information by an optimisation over distributions that preserve the source-target marginals (Bertschinger et al., 2014). The measures can disagree, and no single definition is accepted as correct for all purposes. The quantity also depends on how the target and sources are chosen, so it is a property of a particular decomposition and not of the joint distribution alone.

Redundancy is distinct from, though related to, co-information, whose positive and negative values indicate net redundancy or net synergy.

## Redundancy in coding
Shannon's earlier notion defines redundancy of a source as $1-H/H_{\max}$, the fraction of its capacity not used to carry information, which error-correcting codes add deliberately to resist noise. Barlow (1961) proposed that sensory systems reduce redundancy in their input to represent it efficiently. The two senses are related but not identical.

## Relation to Active Inference
Redundant information is a tool for analysing multivariate information flow, for example how much of the information that sensory and active states carry about internal states is shared. It is not part of the free energy principle. The link is that of an analytic method applied to systems modelled with Active Inference, and results depend on the chosen redundancy measure.

## Related concepts
- [[information_decomposition]] — the framework defining redundancy
- [[synergistic_information]] — information available only jointly
- [[unique_information]] — information available from one source
- [[mutual_information]] — the total being decomposed
- [[information_theory]] — foundations of the quantities used

## References
- Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information. arXiv:1004.2515.
- Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014). Quantifying unique information. Entropy, 16, 2161-2183.
- Harder, M., Salge, C., & Polani, D. (2013). Bivariate measure of redundant information. Physical Review E, 87, 012130.
- Barlow, H. B. (1961). Possible principles underlying the transformations of sensory messages. In W. A. Rosenblith (Ed.), Sensory Communication (pp. 217-234). MIT Press.
