---
title: Information Flow
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - information_flow
  - transfer_entropy
  - data_processing_inequality
semantic_relations:
  - type: foundation
    links:
      - "[[information_theory]]"
      - "[[mutual_information]]"
  - type: relates
    links:
      - "[[markov_blanket]]"
      - "[[message_passing]]"
      - "[[causality]]"
aliases:
  - transfer_entropy
---

# Information Flow

## Overview
Information flow quantifies how much one part of a system tells you about, or influences, another over time or along a pathway. Several non-equivalent measures exist: static dependence (mutual information), directed predictive transfer (transfer entropy, Granger causality), and interventional influence (causal information flow).

## Definition / Formalism
Mutual information $I(X;Y)=D_{\mathrm{KL}}[p(x,y)\|p(x)p(y)]$ is symmetric and measures shared information. The data processing inequality states that for a Markov chain $X\to Y\to Z$, $I(X;Z)\le I(X;Y)$: processing cannot create information. Numerical check: for $p(x)=(0.3,0.7)$, $p(y|x)=\begin{pmatrix}0.9&0.1\\0.2&0.8\end{pmatrix}$, $p(z|y)=\begin{pmatrix}0.7&0.3\\0.1&0.9\end{pmatrix}$, one finds $I(X;Y)=0.229$ nats and $I(X;Z)=0.080$ nats.

Schreiber's transfer entropy from $Y$ to $X$ is a conditional mutual information

$$T_{Y\to X}=I\big(X_{t+1};Y_t^{(l)}\,\big|\,X_t^{(k)}\big),$$

the reduction in uncertainty about the next value of $X$ from the history of $Y$ beyond $X$'s own history. It is asymmetric. For jointly Gaussian processes it equals half the Granger causality statistic (Barnett, Barrett and Seth, 2009). Transfer entropy measures predictive information, not causation: a hidden common driver produces nonzero $T$ with no direct influence, and interventional definitions (Ay and Polani, 2008) are needed for causal claims. Estimation from finite data is biased and sensitive to the choice of history lengths.

## Role in Active Inference
A Markov blanket is a statement about information flow: internal and external states are conditionally independent given blanket states, $I(\mu;\eta\mid b)=0$, so any coupling between agent and world passes through sensory and active states. This structural fact, together with the data processing inequality, implies the agent's beliefs about the world are mediated by what crosses the blanket. Message passing on factor graphs is an operational form of directed information flow inside a generative model. Transfer entropy and Granger-type measures are used to analyse neural and behavioural data, including in studies that fit active inference models, but they are descriptive tools and do not follow from the free energy principle.

## Related concepts
- [[mutual_information]] — the symmetric measure underlying these quantities
- [[markov_blanket]] — conditional independence structure that constrains flow
- [[message_passing]] — directed propagation of beliefs in graphical models
- [[causality]] — distinguishing predictive from causal influence
- [[information_theory]] — foundations of entropy and divergence
- [[information_processing]] — cognitive treatment of information transformations

## References
- Schreiber (2000), Measuring information transfer, Physical Review Letters 85.
- Cover & Thomas, Elements of Information Theory.
- Barnett, Barrett & Seth (2009), Granger causality and transfer entropy are equivalent for Gaussian variables, Physical Review Letters 103.
