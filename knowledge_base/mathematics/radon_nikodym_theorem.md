---
title: Radon-Nikodym Theorem
description: The measure-theoretic theorem giving the density of one measure with respect to another under absolute continuity, with applications to likelihood ratios, KL divergence and importance sampling.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - measure_theory
  - probability
  - densities
semantic_relations:
  - type: relates
    links:
      - "[[measure_theory]]"
      - "[[measure_space]]"
      - "[[sigma_algebras]]"
      - "[[kl_divergence]]"
      - "[[importance_sampling]]"
---

# Radon-Nikodym Theorem

## Overview
The Radon-Nikodym theorem identifies when one measure can be written as an integral of a density against another. It supplies the general definition of a probability density, so that statements about likelihoods, divergences and change of measure hold for discrete, continuous and mixed distributions alike.

## Statement
Let $(X,\mathcal{F})$ be a measurable space and let $\mu$ and $\nu$ be $\sigma$-finite measures on it. The measure $\nu$ is absolutely continuous with respect to $\mu$, written $\nu\ll\mu$, if $\mu(A)=0$ implies $\nu(A)=0$. The theorem states that $\nu\ll\mu$ if and only if there exists a measurable function $f:X\to[0,\infty)$ such that
$$\nu(A)=\int_Af\,d\mu\qquad\text{for all }A\in\mathcal{F}.$$
The function $f=\dfrac{d\nu}{d\mu}$ is the Radon-Nikodym derivative and is unique up to $\mu$-null sets. Two rules follow from the definition: the chain rule $\dfrac{d\nu}{d\lambda}=\dfrac{d\nu}{d\mu}\dfrac{d\mu}{d\lambda}$ when $\nu\ll\mu\ll\lambda$, and $\int g\,d\nu=\int g\,\dfrac{d\nu}{d\mu}\,d\mu$ for integrable $g$. The $\sigma$-finiteness assumption cannot be dropped: counting measure on $[0,1]$ dominates Lebesgue measure there without any density relating them. For a signed measure of finite variation, the Lebesgue decomposition splits it into an absolutely continuous part and a singular part.

## Uses in probability and inference
- **Densities.** If $\mu$ is Lebesgue measure and $\nu$ a probability law with $\nu\ll\mu$, then $f$ is the usual density. If $\mu$ is counting measure, $f$ is the probability mass function.
- **Likelihood ratios.** For laws $P\ll Q$, the ratio $dP/dQ$ is the likelihood ratio, which in a sequential setting is a non-negative $Q$-martingale ([[martingale_theory]]).
- **Divergences.** The Kullback-Leibler divergence is defined by $D_{\mathrm{KL}}(P\|Q)=\int\ln\!\big(\tfrac{dP}{dQ}\big)\,dP$ when $P\ll Q$ and is $+\infty$ otherwise ([[kl_divergence]], [[f_divergences]]).
- **Importance sampling.** Expectations under $P$ can be estimated from samples of $Q$ weighted by $dP/dQ$ ([[importance_sampling]]).
- **Change of measure for processes.** Girsanov's theorem expresses the density between the laws of diffusions with different drifts.

## Relation to Active Inference
The theorem underlies the formal statements of the [[free_energy_principle]] in general state spaces. Variational free energy is a functional of probability measures, written with a Radon-Nikodym derivative between the approximate posterior and the generative model, and the requirement $q\ll p$ is necessary for the divergence to be finite. Measure-theoretic treatments of the framework make this explicit ([[measure_theoretic_active_inference]]). For finite-state models used in practice, the theorem reduces to ratios of probability vectors.

## Related concepts
- [[measure_theory]] — the theory in which the theorem is stated
- [[sigma_algebras]] — the sets on which measures are defined
- [[lebesgue_integration]] — integration against a measure
- [[kl_divergence]] — expressed through the derivative
- [[importance_sampling]] — reweighting by the derivative
- [[measure_theoretic_active_inference]] — measure-theoretic formulation of the framework

## References
- Rudin, W. (1987). Real and Complex Analysis (3rd ed.). McGraw-Hill.
- Billingsley, P. (1995). Probability and Measure (3rd ed.). Wiley.
