---
title: Fractal Dimensions
description: Non-integer dimensions that quantify the scaling and space-filling properties of irregular sets and chaotic attractors, including box-counting, Hausdorff and correlation dimensions, with standard examples.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - fractals
  - chaos
  - scaling
semantic_relations:
  - type: relates
    links:
      - "[[chaos_theory]]"
      - "[[lyapunov_exponents]]"
      - "[[dynamical_systems]]"
      - "[[scale_invariance]]"
      - "[[complexity_measures]]"
---

# Fractal Dimensions

## Overview
A fractal dimension is a number, generally not an integer, describing how the detail of a set changes with the scale at which it is examined. Mandelbrot (1982) popularised the term for sets whose structure repeats across scales. Several non-equivalent definitions exist, and they agree for simple self-similar sets but can differ for general ones.

## Definitions
- **Box-counting (Minkowski-Bouligand) dimension.** If $N(\varepsilon)$ is the number of boxes of side $\varepsilon$ needed to cover a bounded set,
$$d_B=\lim_{\varepsilon\to0}\frac{\ln N(\varepsilon)}{\ln(1/\varepsilon)},$$
when the limit exists. It is the one most often estimated from data by fitting a line to a log-log plot.
- **Hausdorff dimension.** Defined through covers by sets of varying diameter and the Hausdorff measure; it is the finest of the standard definitions and satisfies $d_H\le d_B$.
- **Correlation dimension.** From the correlation integral $C(\varepsilon)$, the fraction of pairs of points closer than $\varepsilon$, which scales as $C(\varepsilon)\sim\varepsilon^{D_2}$ (Grassberger and Procaccia, 1983). It is computable from a time series and satisfies $D_2\le d_B$.

For exactly self-similar sets made of $m$ copies scaled by a factor $r$, the similarity dimension is $\ln m/\ln(1/r)$. The middle-thirds Cantor set has dimension $\ln2/\ln3\approx0.631$, the Koch curve $\ln4/\ln3\approx1.262$ and the Sierpinski triangle $\ln3/\ln2\approx1.585$.

## Fractals in dynamics
Strange attractors of chaotic systems typically have fractal geometry; the Lorenz attractor has a dimension slightly above two. The Kaplan-Yorke conjecture relates the attractor's dimension to its Lyapunov exponents. Estimating dimensions from finite, noisy data is delicate, since a log-log slope is easily over-interpreted, and a low estimate does not by itself show deterministic chaos.

## Relation to Active Inference
This is a descriptive link. Fractal and power-law scaling is reported in neural and behavioural time series, and the free energy principle's discussions of attractors and itinerant dynamics use the vocabulary of dynamical systems, in which fractal dimension is one summary statistic. The principle does not predict a particular dimension, and no derivation connects the two.

## Related concepts
- [[chaos_theory]] — setting in which strange attractors with fractal geometry arise
- [[lyapunov_exponents]] — rates of divergence related to attractor dimension
- [[power_laws]] — scaling relations that accompany fractal structure
- [[scale_invariance]] — the symmetry underlying self-similar sets
- [[attractor_dynamics]] — the dynamics whose limit sets may be fractal

## References
- Mandelbrot, B. B. (1982). The Fractal Geometry of Nature. W. H. Freeman.
- Grassberger, P., & Procaccia, I. (1983). Characterization of strange attractors. Physical Review Letters, 50, 346-349.
- Falconer, K. (2014). Fractal Geometry: Mathematical Foundations and Applications (3rd ed.). Wiley.
