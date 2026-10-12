---
title: Stationary Processes
description: Stochastic processes whose statistical properties do not change under shifts in time, in the strict sense of invariant joint distributions and the weak sense of constant mean and lag-dependent covariance, including Markov processes at their invariant density.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - stochastic_processes
  - probability
  - time_series
  - steady_state
aliases:
  - stationary processes
semantic_relations:
  - type: relates
    links:
      - "[[stochastic_processes]]"
      - "[[ergodicity]]"
      - "[[non_equilibrium_steady_state]]"
      - "[[fokker_planck]]"
      - "[[time_series_analysis]]"
      - "[[probability_theory]]"
      - "[[markov_blanket]]"
      - "[[da_costa_2021_bayesian]]"
---

# Stationary Processes

## Overview
A stochastic process $\{X_t\}$ is stationary when its statistical description does not depend on the origin of time. Stationarity is what allows a single long record to say something about the process, and it underlies most of classical time series analysis. It is distinct from a process being constant: a stationary process can fluctuate violently, but the rules governing the fluctuations stay the same.

## Definitions
- **Strict stationarity.** For every $n$, every set of times $t_1,\dots,t_n$ and every shift $h$, the joint distribution of $(X_{t_1+h},\dots,X_{t_n+h})$ equals that of $(X_{t_1},\dots,X_{t_n})$.
- **Weak (wide-sense, covariance) stationarity.** The mean $\mathbb{E}[X_t]=\mu$ is constant, the variance is finite, and the autocovariance $\gamma(h)=\operatorname{Cov}(X_t,X_{t+h})$ depends only on the lag $h$. Strict stationarity with finite second moments implies weak stationarity. The converse fails in general, but holds for Gaussian processes, which are determined by their means and covariances.

## Markov processes and invariant densities
For a Markov process, such as the solution of a stochastic differential equation $dx = f(x)\,dt+\sigma\,dW_t$, a probability density $p^*$ is invariant (stationary) if it is unchanged by the dynamics. In terms of the Fokker-Planck equation (see [[fokker_planck]]) this means $\partial_t p^*=0$. A Markov process started from $p^*$ is strictly stationary. The invariant density can describe equilibrium, where probability flows balance in detail and there is no net current, or a non-equilibrium steady state, where a non-zero probability current circulates while the density itself stays fixed (see [[non_equilibrium_steady_state]]).

## Stationarity and ergodicity
Stationarity is not ergodicity. A stationary process is ergodic if averages over time converge to averages over the stationary distribution, so a long time series from one realisation is representative (see [[ergodicity]]). A stationary process that is a mixture of distinct invariant behaviours is not ergodic.

## Uses
In time series analysis one tests for stationarity, differences or detrends a series that is not stationary, and fits models such as autoregressive moving-average processes to the stationary remainder. In statistical physics and the theory of random dynamical systems, the stationary density characterises the long-time behaviour of the system.

## Relation to Active Inference
Stationarity plays a foundational role in one formulation of the framework. Da Costa, Friston, Heins and Pavliotis (2021) assume that a system with a Markov blanket is at steady state, that is, that its states have a stationary density, and from this derive a Bayesian mechanics in which internal states appear to perform inference about external states and active states to perform active inference (see [[markov_blanket]] and [[da_costa_2021_bayesian]]). The result is conditional on that assumption. Systems far from a steady state, or whose statistics drift, fall outside this derivation.

## References
- Da Costa, L., Friston, K., Heins, C., & Pavliotis, G. A. (2021). Bayesian mechanics for stationary processes. Proceedings of the Royal Society A, 477(2256), 20210518. doi:10.1098/rspa.2021.0518
- Doob, J. L. (1953). Stochastic Processes. Wiley.
- Brockwell, P. J., & Davis, R. A. (1991). Time Series: Theory and Methods (2nd ed.). Springer.
- Grimmett, G. R., & Stirzaker, D. R. (2001). Probability and Random Processes (3rd ed.). Oxford University Press.
