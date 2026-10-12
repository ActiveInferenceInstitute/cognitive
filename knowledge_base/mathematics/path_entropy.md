---
title: Path Entropy
description: The Shannon entropy of a probability distribution over trajectories, its entropy-rate form for Markov processes, and its use in maximum caliber and stochastic thermodynamics.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - stochastic_processes
  - maximum_caliber
semantic_relations:
  - type: relates
    links:
      - "[[entropy]]"
      - "[[path_integral]]"
      - "[[maximum_entropy_principle]]"
      - "[[entropy_production]]"
      - "[[stochastic_processes]]"
---

# Path Entropy

## Overview
Path entropy is the Shannon entropy applied to trajectories instead of single states. If $P(\Gamma)$ is a probability distribution over paths $\Gamma=(x_0,x_1,\dots,x_T)$ of a system, then
$$S[P]=-\sum_\Gamma P(\Gamma)\ln P(\Gamma),$$
with the sum replaced by a path-space integral against a reference measure for continuous-time processes. It measures the uncertainty about which whole history occurs, which can exceed the uncertainty about the state at any single time.

## Entropy rate and relative path entropy
For a stationary Markov chain with transition probabilities $p_{ij}$ and stationary distribution $\pi$, path entropy grows linearly in the number of steps, and the entropy rate is
$$h=-\sum_i\pi_i\sum_j p_{ij}\ln p_{ij}.$$
This is the standard information-theoretic quantity described by Cover and Thomas. For a joint path distribution, the chain rule gives $S=\sum_t H(x_t\mid x_{t-1},\dots,x_0)$.

Relative path entropy, the Kullback-Leibler divergence between two path measures $P$ and $Q$, is
$$D_{\mathrm{KL}}(P\|Q)=\sum_\Gamma P(\Gamma)\ln\frac{P(\Gamma)}{Q(\Gamma)}.$$
In stochastic thermodynamics, the divergence between the forward path distribution and that of the time-reversed process measures irreversibility and is linked to total entropy production (Seifert, 2012). See [[entropy_production]] and [[kl_divergence]].

## Maximum caliber
Jaynes proposed extending the maximum entropy principle from states to paths. Maximising $S[P]$ subject to constraints on path-averaged quantities, such as mean currents, yields an exponential-family distribution over trajectories, in analogy with the Gibbs distribution over states. This approach is called maximum caliber, and Presse and colleagues review its applications in non-equilibrium statistical physics ([[maximum_entropy_principle]]).

## Relation to Active Inference
Path entropy is central to the formal treatment of the [[free_energy_principle]], where entropy is taken over the paths of a system's internal and blanket states in addition to its instantaneous states. Expected free energy is likewise defined for policies, which are distributions over future trajectories, and path-based formulations of control relate cost to the divergence of controlled from uncontrolled path measures ([[path_integral_control]], [[path_integral_free_energy]]). These connections are a recognised and active theoretical programme. They are formal links, so interpreting path entropy as a quantity that biological systems literally minimise remains contested.

## Related concepts
- [[entropy]] — the state-level quantity extended to paths
- [[path_integral]] — integration over the same trajectory space
- [[maximum_entropy_principle]] — the inference principle behind maximum caliber
- [[entropy_production]] — irreversibility measured by relative path entropy
- [[stochastic_processes]] — the processes whose paths are considered
- [[kl_divergence]] — the divergence between path measures

## References
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
- Jaynes, E. T. (1980). The minimum entropy production principle. Annual Review of Physical Chemistry, 31, 579-601.
- Presse, S., Ghosh, K., Lee, J., & Dill, K. A. (2013). Principles of maximum entropy and maximum caliber in statistical physics. Reviews of Modern Physics, 85, 1115-1141.
- Seifert, U. (2012). Stochastic thermodynamics, fluctuation theorems and molecular machines. Reports on Progress in Physics, 75, 126001.
