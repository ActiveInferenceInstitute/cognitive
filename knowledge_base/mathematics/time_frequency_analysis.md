---
title: Time-Frequency Analysis
description: Methods that describe how the spectral content of a signal changes over time, including the short-time Fourier transform, wavelets and the Gabor uncertainty bound, with uses for neural oscillations.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - signal_processing
  - spectral_analysis
  - neural_oscillations
semantic_relations:
  - type: relates
    links:
      - "[[signal_processing]]"
      - "[[spectral_theory]]"
      - "[[time_series_analysis]]"
      - "[[neural_synchrony]]"
      - "[[calculus]]"
---

# Time-Frequency Analysis

## Overview
The Fourier transform $\hat x(\omega)=\int x(t)e^{-i\omega t}\,dt$ tells which frequencies a signal contains but not when they occur. Time-frequency analysis represents a signal as a function of both time and frequency, which is needed whenever the spectrum is non-stationary, as in speech, music and neural recordings.

## Main transforms
- **Short-time Fourier transform.** A window $w$ slides along the signal:
$$X(t,\omega)=\int x(\tau)\,w(\tau-t)\,e^{-i\omega\tau}\,d\tau .$$
The squared magnitude $|X(t,\omega)|^2$ is the spectrogram. Window length fixes one resolution for all frequencies.
- **Continuous wavelet transform.** With a mother wavelet $\psi$, scale $a>0$ and shift $b$,
$$W(a,b)=\frac{1}{\sqrt a}\int x(t)\,\overline{\psi\!\left(\frac{t-b}{a}\right)}\,dt .$$
Short windows are used at high frequency and long windows at low frequency. The Morlet wavelet, a Gaussian-windowed complex sinusoid, is common in electrophysiology.
- **Quadratic distributions.** The Wigner-Ville distribution and related members of Cohen's class have sharper concentration for some signals but produce interference terms for signals with several components.

## Uncertainty principle
Time and frequency resolution cannot both be made arbitrarily fine. For a unit-energy signal with time spread $\sigma_t$ and angular-frequency spread $\sigma_\omega$, defined as standard deviations of $|x(t)|^2$ and $|\hat x(\omega)|^2/2\pi$,
$$\sigma_t\,\sigma_\omega\ge\tfrac12,$$
with equality for Gaussians. Gabor (1946) introduced this bound for communication signals, and it sets the trade-off in every transform above. Reducing the window in time necessarily broadens it in frequency.

## Relation to Active Inference
Time-frequency analysis is a measurement tool, not part of the theory. It connects to Active Inference through empirical work: dynamic causal modelling can be applied to spectral features of electrophysiological data, and some predictive coding proposals associate gamma-band activity with prediction errors and lower-frequency activity with predictions (Bastos et al., 2012). These are hypotheses under test, and observed frequency-band assignments vary across cortical areas and tasks.

## Related concepts
- [[signal_processing]] — the broader field of analysing and transforming signals
- [[spectral_theory]] — the operator-theoretic view of decomposition into frequencies
- [[time_series_analysis]] — statistical modelling of temporal data
- [[neural_synchrony]] — a phenomenon commonly studied with time-frequency methods
- [[calculus]] — the integrals and transforms on which the definitions rely

## References
- Gabor, D. (1946). Theory of communication. Journal of the Institution of Electrical Engineers, 93(III), 429-457.
- Cohen, L. (1995). Time-Frequency Analysis. Prentice Hall.
- Mallat, S. (2009). A Wavelet Tour of Signal Processing: The Sparse Way (3rd ed.). Academic Press.
- Bastos, A. M., Usrey, W. M., Adams, R. A., Mangun, G. R., Fries, P., & Friston, K. J. (2012). Canonical microcircuits for predictive coding. Neuron, 76, 695-711.
