---
title: Data Processing Inequality
description: The result that no processing of a signal can increase the information it carries about its source, with a short proof, extensions to divergences, and relevance to Markov blankets and representation learning.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - markov_chain
  - inequality
semantic_relations:
  - type: relates
    links:
      - "[[mutual_information]]"
      - "[[conditional_mutual_information]]"
      - "[[markov_blanket]]"
      - "[[information_bottleneck]]"
      - "[[f_divergences]]"
---

# Data Processing Inequality

## Overview
The data processing inequality (DPI) states that if random variables form a Markov chain $X\to Y\to Z$, meaning $Z$ is conditionally independent of $X$ given $Y$, then
$$
I(X;Z)\le I(X;Y).
$$
Whatever is done to $Y$ to produce $Z$, whether deterministic transformation, noise, quantisation or any other stochastic channel that does not use $X$ again, cannot add information about $X$. Post-processing can discard information or make it easier to use, but not create it.

## Proof sketch
Apply the chain rule for mutual information in two ways:
$$
I(X;Y,Z)=I(X;Y)+I(X;Z\mid Y)=I(X;Z)+I(X;Y\mid Z).
$$
The Markov property gives $I(X;Z\mid Y)=0$, and $I(X;Y\mid Z)\ge0$, so $I(X;Y)=I(X;Z)+I(X;Y\mid Z)\ge I(X;Z)$. Equality holds exactly when $X\to Z\to Y$ is also a Markov chain, which is the case when $Z$ is a sufficient statistic of $Y$ for $X$.

## Extensions and uses
- **Divergences**: the same monotonicity holds for KL divergence and other $f$-divergences, since applying the same channel to two distributions cannot increase the divergence between them.
- **Deterministic post-processing**: a deterministic function $g(Y)$ carries no more information about $X$ than $Y$ does.
- **Representation learning**: in the information bottleneck, successive layers of a feed-forward network form a Markov chain, so information about the input or label cannot increase with depth.
- **Estimation limits**: it underlies converse results in coding theory and bounds on what any estimator can extract from processed data.

## Relation to Active Inference
The connection is a consequence of the conditional independence structure used in the free energy principle. If internal states $\mu$ are conditionally independent of external states $\eta$ given the blanket states $b$, the chain $\eta\to b\to\mu$ holds, so $I(\eta;\mu)\le I(\eta;b)$: whatever the internal states encode about the environment passes through the blanket. This justifies the view of beliefs as shaped by sensory evidence only. The DPI itself is a general theorem in information theory and does not by itself assert the free energy principle, and whether real systems satisfy the blanket assumption exactly is a separate, debated question.

## Related concepts
- [[mutual_information]] — the quantity that cannot increase under processing
- [[conditional_mutual_information]] — vanishes for the middle-conditioned chain
- [[markov_blanket]] — supplies the conditional independence in the Active Inference case
- [[information_bottleneck]] — uses the inequality to analyse compression
- [[f_divergences]] — the family for which the same monotonicity holds
- [[channel_capacity]] — limit on information passed through a noisy channel

## References
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
- MacKay, D. J. C. (2003). Information Theory, Inference, and Learning Algorithms. Cambridge University Press.
