---
title: Signal Processing
description: "Sampling, Fourier analysis, linear filtering and Kalman filtering as the engineering mathematics of extracting signals from noise, and its relation to predictive coding and generalised filtering."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - signal_processing
  - filtering
  - fourier
semantic_relations:
  - type: relates
    links:
      - "[[predictive_coding]]"
      - "[[calculus]]"
      - "[[stochastic_processes]]"
      - "[[neural_computation]]"
---

# Signal Processing

## Overview
Signal processing analyses, transforms and estimates signals, which are functions of time or space, often corrupted by noise. Its core tools are Fourier analysis, linear time-invariant (LTI) filtering, and recursive state estimation.

## Core tools
**Sampling.** A band-limited signal with maximum frequency $B$ is determined by samples at rate above $2B$ (Nyquist-Shannon); undersampling causes aliasing.

**Fourier transform and convolution.** An LTI system with impulse response $h$ maps input $x$ to $y=x*h$, and $\hat y=\hat x\,\hat h$. I checked the discrete version: FFT products with zero padding reproduce `np.convolve` exactly. Filters are designed by shaping $\hat h(\omega)$ (low-pass, band-pass), and the uncertainty relation limits joint time and frequency resolution.

**Wiener filter.** The linear filter minimising mean squared error between an estimate and a stationary target, given known spectra: $H(\omega)=S_{sx}(\omega)/S_{xx}(\omega)$, the cross-spectrum of target and observation over the observation spectrum.

**Kalman filter.** For $x_{t+1}=ax_t+w_t$, $y_t=x_t+v_t$ with Gaussian noise variances $q,r$, the posterior is Gaussian and updated recursively by predict and correct steps with gain $K=P^-/(P^-+r)$. For $a=0.9,q=0.1,r=0.5$ the gain converges to about 0.311. In continuous time the equivalent is the Kalman-Bucy filter. The Kalman filter is exact Bayesian filtering for linear Gaussian systems.

**Time-frequency methods.** Short-time Fourier and wavelet transforms handle non-stationary signals, which is the practical regime for neural data.

## Role in Active Inference
Signal processing supplies both data analysis tools (spectral analysis of EEG/MEG, dynamic causal modelling in frequency space) and the dynamical form of inference. [[predictive_coding]] can be viewed as a nonlinear generalisation of Kalman filtering with hierarchical generative models, and Friston's variational filtering and generalised filtering work in generalised coordinates ([[generalized_coordinates]]) to estimate states in continuous time while treating noise as smooth rather than white. These schemes differ from the Kalman-Bucy filter chiefly in assuming smooth rather than white noise, so they are related to it but not equivalent. Discrete-state Active Inference uses little of this machinery.

## Related concepts
- [[predictive_coding]] — hierarchical filtering in generative models
- [[generalized_coordinates]] — representation of temporal derivatives as states
- [[stochastic_processes]] — noise models for signals
- [[calculus]] — transforms and differential operators
- [[neural_computation]] — signal transformation in neural circuits

## References
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. Journal of Basic Engineering.
- Friston, K., Stephan, K., Li, B., Daunizeau, J. (2010). Generalised filtering. Mathematical Problems in Engineering.
