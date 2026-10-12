---
title: Information Projection
description: The minimisation of KL divergence from a distribution to a constrained family, in its two forms, I-projection and M-projection, with the Pythagorean identity and its role in variational inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_geometry
  - kl_divergence
  - variational_inference
semantic_relations:
  - type: relates
    links:
      - "[[kl_divergence]]"
      - "[[information_geometry]]"
      - "[[variational_inference]]"
      - "[[exponential_family]]"
      - "[[maximum_entropy_principle]]"
wikidata: Q6031173
---

# Information Projection

## Overview
An information projection finds the member of a family $\mathcal{Q}$ of distributions that is closest, in the sense of KL divergence, to a reference distribution $p$. There are two versions because the divergence is not symmetric:
$$
q_{\mathrm{I}}=\arg\min_{q\in\mathcal{Q}}D_{\mathrm{KL}}(q\,\|\,p),\qquad q_{\mathrm{M}}=\arg\min_{q\in\mathcal{Q}}D_{\mathrm{KL}}(p\,\|\,q).
$$
The first is the I-projection (information or reverse projection) and the second the M-projection (moment projection). Csiszár (1975) developed the I-divergence geometry of such minimisation problems.

## Properties
- **Pythagorean identity**: if $\mathcal{Q}$ is a convex set of distributions and $q^{*}$ is the I-projection of $p$ onto it, then $D_{\mathrm{KL}}(r\,\|\,p)\ge D_{\mathrm{KL}}(r\,\|\,q^{*})+D_{\mathrm{KL}}(q^{*}\,\|\,p)$ for every $r\in\mathcal{Q}$, with equality when $\mathcal{Q}$ is defined by linear (moment) constraints. This mirrors the Pythagorean theorem for Euclidean projection.
- **Maximum entropy**: the I-projection of a uniform reference distribution onto a set defined by moment constraints is the maximum entropy distribution consistent with them, an exponential family member.
- **Moment matching**: projecting $p$ onto an exponential family by the M-projection yields the member whose expected sufficient statistics equal those of $p$.
- **Behaviour**: the I-projection tends to be mode-seeking, placing $q$ where $p$ has mass and avoiding regions where $p$ is small, so for a multimodal $p$ it often captures one mode. The M-projection is mass-covering and tends to spread over all modes.

In the language of information geometry, these correspond to projections along dually flat structures, with the two divergence orderings tied to the mixture and exponential connections (Amari & Nagaoka, 2000).

## Relation to Active Inference
Variational inference, and so variational free energy minimisation, is an I-projection. The free energy differs from $D_{\mathrm{KL}}[q(s)\,\|\,p(s\mid o)]$ only by the constant $\ln p(o)$, so minimising it over a tractable family such as a mean-field family projects the true posterior onto that family. This explains the characteristic behaviour of variational beliefs: they tend to underestimate posterior uncertainty and to lock onto one mode. Expectation propagation uses M-projections locally instead. The identity is exact for the quantity being minimised; whether biological inference uses this particular direction is a modelling assumption.

## Related concepts
- [[kl_divergence]] — the divergence being minimised
- [[information_geometry]] — the geometric setting for the projections
- [[variational_inference]] — the application that performs I-projection
- [[exponential_family]] — families where M-projection reduces to moment matching
- [[maximum_entropy_principle]] — a special case of I-projection
- [[moment_matching]] — the condition satisfied by the M-projection

## References
- Csiszár, I. (1975). I-divergence geometry of probability distributions and minimization problems. Annals of Probability, 3, 146-158.
- Amari, S., & Nagaoka, H. (2000). Methods of Information Geometry. American Mathematical Society and Oxford University Press.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
