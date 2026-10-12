---
title: Information Decomposition
description: Partial information decomposition, which splits the information that several source variables carry about a target into redundant, unique and synergistic parts, and the open problem of how to measure them.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - partial_information_decomposition
  - multivariate_information
semantic_relations:
  - type: relates
    links:
      - "[[mutual_information]]"
      - "[[redundant_information]]"
      - "[[synergistic_information]]"
      - "[[unique_information]]"
      - "[[conditional_mutual_information]]"
---

# Information Decomposition

## Overview
Shannon's mutual information $I(T;X_1,X_2)$ says how much two sources $X_1,X_2$ jointly tell about a target $T$, but not where that information comes from. Information decomposition, introduced as partial information decomposition (PID) by Williams and Beer (2010), divides the total into non-negative parts according to which sources supply it.

## Definition and structure
For two sources the decomposition is
$$I(T;X_1,X_2)=\mathrm{Red}+\mathrm{Unq}_1+\mathrm{Unq}_2+\mathrm{Syn},$$
with the consistency conditions $I(T;X_1)=\mathrm{Red}+\mathrm{Unq}_1$ and $I(T;X_2)=\mathrm{Red}+\mathrm{Unq}_2$. Redundancy is information available from either source alone, unique information is available from one source only, and synergy is information present only when both are known together. The classical quantities supply three equations for four unknowns, so a further definition of one part, usually redundancy, is required to fix the rest. Combining the equations shows that the co-information $I(T;X_1)+I(T;X_2)-I(T;X_1,X_2)$ equals $\mathrm{Red}-\mathrm{Syn}$, so it conflates the two.

Standard examples fix intuition. If $T=X_1\oplus X_2$ for independent fair bits, each source alone carries nothing and the one bit of information is purely synergistic. If both sources are copies of $T$, the information is purely redundant.

For more than two sources, Williams and Beer organised the possible redundancies into a lattice and proposed a measure $I_{\min}$ based on specific information. Later work argued that $I_{\min}$ measures only the amount of overlap rather than whether sources carry the same information, and proposed alternatives (Bertschinger et al., 2014; Harder et al., 2013). No single measure has been accepted universally, and different measures can give different values for the same distribution.

## Role in Active Inference
Decomposition tools are used to analyse multivariate information flow in neural and agent-based models, for example how sensory and active states jointly constrain internal states across a Markov blanket. This is a descriptive use: PID is not a component of the free energy principle and gives no derivation of it. The strength of the link is that of an analysis method applied to systems that Active Inference models, with the caveat that conclusions depend on the chosen redundancy measure.

## Related concepts
- [[mutual_information]] — the quantity being decomposed
- [[redundant_information]] — shared component available from any source
- [[synergistic_information]] — component available only from sources jointly
- [[unique_information]] — component available from one source alone
- [[conditional_mutual_information]] — its conditional form equals unique plus synergistic information
- [[markov_blankets]] — a setting where multivariate information flow is analysed

## References
- Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information. arXiv:1004.2515.
- Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014). Quantifying unique information. Entropy, 16, 2161-2183.
- Harder, M., Salge, C., & Polani, D. (2013). Bivariate measure of redundant information. Physical Review E, 87, 012130.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
