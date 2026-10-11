---
title: Jensen-Shannon Divergence
description: The symmetric, bounded divergence defined as the average KL divergence of two distributions to their mixture, its metric square root, and its limited role relative to KL in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - divergences
  - information_theory
  - entropy
semantic_relations:
  - type: relates
    links:
      - "[[kl_divergence]]"
      - "[[f_divergences]]"
      - "[[entropy]]"
      - "[[mutual_information]]"
      - "[[jensen_inequality]]"
---

# Jensen-Shannon Divergence

## Overview
The Jensen-Shannon divergence (JSD) measures the dissimilarity of two probability distributions $P$ and $Q$. It is a smoothed, symmetrised version of the Kullback-Leibler divergence, built by comparing each distribution to their equal mixture $M=\tfrac12(P+Q)$:

$$\mathrm{JSD}(P\|Q)=\tfrac12 D_{\mathrm{KL}}(P\|M)+\tfrac12 D_{\mathrm{KL}}(Q\|M).$$

It was introduced in this form for information-theoretic applications by Lin (1991).

## Properties
- **Symmetric**: $\mathrm{JSD}(P\|Q)=\mathrm{JSD}(Q\|P)$, unlike [[kl_divergence]].
- **Always finite**: because $P$ and $Q$ are both absolutely continuous with respect to $M$, the divergence is defined even when their supports differ, whereas $D_{\mathrm{KL}}(P\|Q)$ is infinite in that case.
- **Bounded**: $0\le\mathrm{JSD}\le\ln 2$ in nats, or $\le 1$ in bits, with $0$ only for $P=Q$. The maximum occurs for distributions with disjoint supports.
- **Entropy form**: $\mathrm{JSD}(P\|Q)=H(M)-\tfrac12H(P)-\tfrac12H(Q)$, the entropy of the mixture minus the average entropy of the components. Non-negativity is therefore an instance of concavity of entropy ([[jensen_inequality]]).
- **Mutual information reading**: it equals the mutual information between a variable $X$ drawn from the mixture and the binary label indicating which component generated it.
- **Metric property**: $\sqrt{\mathrm{JSD}}$ satisfies the triangle inequality and is a true metric on distributions. The generalisation to more than two distributions uses weights and is likewise a difference of entropies.

JSD is also an f-divergence, with a specific convex generator (see [[f_divergences]]).

## Use and relation to Active Inference
JSD appears in practice as a symmetric distance between empirical distributions, for example in comparing word frequencies, genetic sequences or model outputs, and in the original generative adversarial network objective, whose optimal discriminator turns the loss into a JSD between data and model distributions. Active Inference is built on the asymmetric KL divergence instead: variational free energy contains the KL from approximate to true posterior, and expected free energy decomposes into expected information gain and expected cost, both KL-based. JSD is therefore a neighbouring tool, handy for comparing beliefs or distributions across agents or time, rather than a quantity the framework minimises. The connection is one of shared information-theoretic machinery, not a derivation.

## Related concepts
- [[kl_divergence]] — the asymmetric divergence from which JSD is built
- [[f_divergences]] — the family that contains JSD
- [[entropy]] — JSD is a difference of entropies
- [[mutual_information]] — JSD as information about the component label
- [[jensen_inequality]] — gives non-negativity through concavity of entropy

## References
- Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37, 145-151.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
