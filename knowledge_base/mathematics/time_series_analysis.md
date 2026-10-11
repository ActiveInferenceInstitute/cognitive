---
title: Time Series Analysis
description: Methods for modelling and forecasting ordered observations over time, including stationarity, autocorrelation, ARMA models, spectral analysis and state-space formulations, and their use in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - time_series
  - stochastic_processes
  - state_space
semantic_relations:
  - type: relates
    links:
      - "[[stochastic_processes]]"
      - "[[state_space_models]]"
      - "[[signal_processing]]"
      - "[[granger_causality]]"
      - "[[dynamical_systems]]"
---

# Time Series Analysis

## Overview
A time series is a sequence of observations $x_1,\dots,x_T$ indexed by time, typically with dependence between neighbouring values. Time series analysis describes that dependence, forecasts future values, and separates structure such as trend, seasonality and noise. Because successive observations are not independent, the usual assumptions of independent samples fail and dedicated methods are required.

## Core concepts
- **Stationarity.** A process is weakly stationary if its mean is constant and its covariance depends only on the lag $h$. Many methods require stationarity, which can often be approached by differencing or detrending.
- **Autocorrelation.** The function $\rho(h)=\mathrm{Corr}(x_t,x_{t+h})$ summarises linear dependence across lags and guides model choice.
- **ARMA and ARIMA models.** An autoregressive moving-average model writes $x_t=\sum_{i=1}^{p}\phi_i x_{t-i}+\varepsilon_t+\sum_{j=1}^{q}\theta_j \varepsilon_{t-j}$ with white-noise innovations $\varepsilon_t$. The ARIMA extension adds differencing for non-stationary series.
- **Spectral analysis.** The Fourier transform of the autocovariance gives the power spectrum, which shows how variance is distributed over frequencies.
- **State-space models.** A latent state $z_t$ evolves by a transition model and emits observations $x_t$. For linear Gaussian models the Kalman filter computes the exact filtering distribution recursively, and nonlinear cases use approximations such as particle filtering.

## Relation to Active Inference
The connection is structural. The generative models of Active Inference are state-space models: hidden states evolve under a transition model, and observations are generated from them (see [[state_space_models]] and [[generalized_coordinates]]). Perception then amounts to filtering a time series of sensory data, and in continuous-time formulations the filter is expressed in generalised coordinates of motion. Time series methods are also used on the data side, for instance when models of neural activity are fitted to electrophysiological or imaging recordings and when directed influence between regions is assessed. Time series analysis is a general methodology, and Active Inference is one specific family of generative models to which it applies.

## Related concepts
- [[stochastic_processes]] — the probabilistic objects that time series are realisations of
- [[state_space_models]] — latent-state formulation used for filtering
- [[signal_processing]] — shares filtering and spectral tools
- [[granger_causality]] — predictive notion of directed influence between series
- [[dynamical_systems]] — deterministic counterpart for the underlying dynamics
- [[particle_filtering]] — sequential Monte Carlo inference for nonlinear state-space models

## References
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control (5th ed.). Wiley.
- Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press.
