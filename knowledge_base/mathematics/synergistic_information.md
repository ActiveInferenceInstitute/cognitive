---
title: Synergistic Information
description: The part of the information that several sources carry about a target only jointly and not in any source alone, as formalised in partial information decomposition, with the XOR example.
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
      - "[[unique_information]]"
      - "[[redundant_information]]"
      - "[[mutual_information]]"
      - "[[integrated_information]]"
---

# Synergistic Information

## Overview
When two or more source variables inform about a target, the total information they provide is not simply the sum of what each provides alone. Synergistic information is the component available only from the sources taken together. It is one term of the partial information decomposition (PID) of Williams and Beer (2010), alongside redundant and unique information.

## Decomposition
For a target $T$ and two sources $X$ and $Y$, the joint mutual information splits into four non-negative parts:
$$I(T;X,Y)=\mathrm{Red}(T;X,Y)+\mathrm{Unq}(T;X\setminus Y)+\mathrm{Unq}(T;Y\setminus X)+\mathrm{Syn}(T;X,Y).$$
The single-source terms satisfy $I(T;X)=\mathrm{Red}+\mathrm{Unq}_X$ and $I(T;Y)=\mathrm{Red}+\mathrm{Unq}_Y$. Subtracting gives the interaction information $I(T;X)+I(T;Y)-I(T;X,Y)=\mathrm{Red}-\mathrm{Syn}$, so classical measures only report the net balance of redundancy and synergy. The conditional mutual information satisfies $I(T;X\mid Y)=\mathrm{Unq}_X+\mathrm{Syn}$.

The standard example is the exclusive-or. If $X,Y$ are independent fair bits and $T=X\oplus Y$, then $I(T;X)=I(T;Y)=0$ while $I(T;X,Y)=1$ bit, so all the information is synergistic. Neither bit alone says anything about the parity.

## Status of the definition
The three identities above (for the joint and the two single-source informations) do not determine the four terms, so a further definition of one of them is required. Williams and Beer proposed a measure of redundancy ($I_{\min}$) that has been criticised for counting some quantities as shared that are not the same information. Bertschinger and colleagues defined unique information through an optimisation over distributions that preserve the source-target marginals, and the synergy then follows. Different definitions give different values, and the extension to more than two sources has not reached a consensus form.

## Relation to Active Inference
The connection is an analogy and a tool for analysis. Synergy has been used to describe how neural populations or system components jointly encode quantities, and it is conceptually adjacent to measures such as [[integrated_information]] that ask how much a whole exceeds its parts. The free energy principle does not itself define a synergy term, and any application to Markov blanket partitions is a research question, not a result.

## Related concepts
- [[information_decomposition]] — the general framework of splitting information into parts
- [[unique_information]] — information held by one source and not the other
- [[redundant_information]] — information that sources share
- [[mutual_information]] — the quantity being decomposed
- [[integrated_information]] — a different whole-versus-parts measure

## References
- Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information. arXiv:1004.2515.
- Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014). Quantifying unique information. Entropy, 16, 2161-2183.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
