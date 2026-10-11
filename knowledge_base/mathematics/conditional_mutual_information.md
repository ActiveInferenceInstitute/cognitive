---
title: Conditional Mutual Information
description: The information shared by two random variables after accounting for a third, its chain rule, its use in testing conditional independence and defining transfer entropy, and its role in epistemic value.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - mutual_information
  - conditional_independence
semantic_relations:
  - type: relates
    links:
      - "[[mutual_information]]"
      - "[[conditional_entropy]]"
      - "[[conditional_independence]]"
      - "[[information_gain]]"
      - "[[information_flow]]"
---

# Conditional Mutual Information

## Overview
The conditional mutual information $I(X;Y\mid Z)$ measures how much knowing $Y$ reduces uncertainty about $X$ when $Z$ is already known:
$$
I(X;Y\mid Z)=H(X\mid Z)-H(X\mid Y,Z)=\mathbb{E}_{z}\Big[D_{\mathrm{KL}}\big(p(x,y\mid z)\,\|\,p(x\mid z)\,p(y\mid z)\big)\Big].
$$
It is symmetric in $X$ and $Y$, always non-negative, and equal to zero if and only if $X$ and $Y$ are conditionally independent given $Z$.

## Properties
- **Chain rule**: $I(X;Y,Z)=I(X;Z)+I(X;Y\mid Z)$, which allows total information to be accumulated variable by variable.
- **Conditioning can raise or lower information**: it can be larger or smaller than $I(X;Y)$. If $X$ and $Y$ are independent fair bits and $Z=X\oplus Y$, then $I(X;Y)=0$ but $I(X;Y\mid Z)=1$ bit, so conditioning on a common effect creates dependence.
- **Common cause**: if $Z$ is a common cause that makes $X$ and $Y$ dependent, conditioning on $Z$ removes that dependence and can leave $I(X;Y\mid Z)=0$.
- **Data processing**: $X\to Z\to Y$ forming a Markov chain is equivalent to $I(X;Y\mid Z)=0$.

Estimating it from finite data is harder than estimating mutual information, because conditioning on a high-dimensional $Z$ requires many samples. Applications include screening for conditional independence in causal discovery and defining transfer entropy, the conditional mutual information between the past of one process and the present of another given the past of the second (Schreiber, 2000), which is used as a directed, model-free measure of information flow.

## Relation to Active Inference
The link is a derivation. The epistemic value of a policy $\pi$ in expected free energy is the expected information gain about hidden states, which equals the mutual information between states and outcomes under that policy, $I_{q}(s;o\mid\pi)$. This is a conditional mutual information in which the policy is the conditioning variable. Transfer entropy and related quantities are also used to analyse directed interactions between neural populations or agents, but these analyses are an application of the measure and are not required by the theory.

## Related concepts
- [[mutual_information]] — the unconditional quantity
- [[conditional_entropy]] — the entropy form from which it is defined
- [[conditional_independence]] — the property it detects when equal to zero
- [[information_gain]] — expected reduction of uncertainty from an observation
- [[information_flow]] — directed transfer of information between processes
- [[granger_causality]] — equivalent to transfer entropy for Gaussian variables

## References
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
- Schreiber, T. (2000). Measuring information transfer. Physical Review Letters, 85, 461-464.
- MacKay, D. J. C. (2003). Information Theory, Inference, and Learning Algorithms. Cambridge University Press.
