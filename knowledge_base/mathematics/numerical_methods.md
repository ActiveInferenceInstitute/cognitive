---
title: Numerical Methods
type: guide
status: stable
created: 2025-03-03
updated: 2026-10-09
tags:
- numerics
- stability
semantic_relations:
  - type: relates_to
    links:
      - "[[softmax_function]]"
      - "[[kl_divergence]]"
      - "[[entropy]]"
aliases:
  - numerical_analysis
---


# Numerical Methods

Active Inference computations are dominated by three operations: normalising vectors and matrices into probabilities, exponentiating and normalising logits (softmax), and taking logarithms of probabilities. Each has a standard failure mode in floating point, and the repository handles each in a specific, inspectable way.

## Stable softmax and log-sum-exp

Policy probabilities are $\mathrm{softmax}(-G/T)$. Evaluated naively, $e^{x_i} / \sum_j e^{x_j}$ overflows for large logits: with $x = (1000, 1001)$, NumPy returns `[nan, nan]`. Subtracting the maximum leaves the result unchanged, because the factor $e^{-\max x}$ cancels, and bounds every exponent by zero. The result for the same input is $(0.2689,\ 0.7311)$. The corresponding log-normaliser is $\ln \sum_j e^{x_j} = m + \ln \sum_j e^{x_j - m}$ with $m = \max_j x_j$, which equals $1001.3133$ here.

`softmax` in `code/tools/src/utils/matrix_utils.py` and `MatrixOps.softmax` in `code/tools/src/models/matrices/matrix_ops.py` both divide by a positive, finite `temperature`, subtract the per-axis maximum, and clip exponents to $[-745, 709]$ (the range over which `np.exp` in `float64` returns a finite, non-zero value). Temperature changes the sharpness smoothly: for $G = (1.1694, 0.7277)$ the second action receives $0.988$, $0.609$ and $0.511$ of the probability at $T = 0.1, 1, 10$.

## Logarithms of probabilities

Entropy and KL divergence need $\ln p$, which diverges at $p = 0$. Two conventions appear in the code.

1. Skip zero terms. Since $\lim_{p \to 0} p \ln p = 0$, an entropy sum can mask out zero entries instead of adding a floor. `_entropy` in `generative_model.py` and `MatrixOps.compute_entropy` do this, so a deterministic distribution has entropy exactly zero rather than a small negative number.
2. Add a floor $\epsilon = 10^{-12}$ inside the logarithm, as in `kl_divergence`, which evaluates $\sum_i p_i \ln\frac{p_i + \epsilon}{q_i + \epsilon}$ over entries with $p_i > 0$.

The floor changes what the answer means when $q$ has zeros: $D_{\mathrm{KL}}\big((0.5, 0.5)\,\|\,(1, 0)\big)$ is infinite mathematically, but the floored function returns $13.12$. Treat a large finite KL as a signal of a zero-probability preference or likelihood entry, not as a calibrated number. Likewise `np.log(np.exp(-800.0) + 1e-12)` is $-27.63$, the floor, because $e^{-800}$ underflows to zero.

## Normalisation and degenerate input

`MatrixOps.normalize_columns` clamps negatives to zero, divides each column by its sum, and replaces a column whose sum is below $10^{-12}$ with the uniform distribution. A column `[0, 0]` therefore becomes `[0.5, 0.5]` rather than `nan`, while `[1, 3]` becomes `[0.25, 0.75]`. This silent fallback is useful in a simulation loop but can hide a model bug, so `DiscreteGenerativeModel` validates `A` and `B` as column stochastic at construction instead of normalising them.

## Conditioning and precision

Precision-weighted updates divide by prediction error: `ActiveInferenceModel.update_precision` computes the target $1/(\text{error} + 10^{-8})$, and `ActiveInferenceDispatcher.update_precision` clips its smoothed value to $[0.1, 10^{6}]$. The additive constant and clip bound the step when the prediction error approaches zero; see [[precision_parameter]]. Single precision is not adequate for these updates: in `float32`, $1 + 10^{-8}$ equals $1$.

## Checklist

- Work in log space for products of many probabilities; exponentiate once, after subtracting the maximum.
- Mask zeros in entropies; floor inside KL and report that the floor is active.
- Validate stochasticity at model construction; normalise only when a fallback is intended.
- Compare against a closed form on a tiny case before trusting a large run.

```python
import numpy as np

from cognitive.models.matrices.matrix_ops import MatrixOps
from cognitive.utils.matrix_utils import kl_divergence, softmax

x = np.array([1000.0, 1001.0])
with np.errstate(all="ignore"):
    print(np.exp(x) / np.exp(x).sum())          # [nan nan]
print(softmax(x))                                # [0.2689 0.7311]
m = x.max()
print(m + np.log(np.exp(x - m).sum()))           # 1001.3133

G = np.array([1.1694, 0.7277])
for temperature in (0.1, 1.0, 10.0):
    print(temperature, softmax(-G, temperature=temperature).round(3))

print(kl_divergence(np.array([0.5, 0.5]), np.array([1.0, 0.0])))  # about 13.12
print(MatrixOps.normalize_columns(np.array([[0.0, 1.0], [0.0, 3.0]])))
print(np.float32(1) + np.float32(1e-8) == np.float32(1))          # True
```

## Links

- [[numerical_stability]]
- [[precision_parameter]]
- [[softmax_function]] · [[kl_divergence]] · [[entropy]]

## References

- Higham, N. J. (2002). Accuracy and stability of numerical algorithms (2nd ed.). Society for Industrial and Applied Mathematics.
- Blanchard, P., Higham, D. J., & Higham, N. J. (2021). Accurately computing the log-sum-exp and softmax functions. IMA Journal of Numerical Analysis, 41(4), 2311-2330.
- Goldberg, D. (1991). What every computer scientist should know about floating-point arithmetic. ACM Computing Surveys, 23(1), 5-48.
- Cover, T. M., & Thomas, J. A. (2006). Elements of information theory (2nd ed.). Wiley.
