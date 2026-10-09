---
title: Matrix Operations
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - linear_algebra
  - stochastic_matrices
  - implementation
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/linear_algebra]]"
      - "[[knowledge_base/mathematics/pomdp_structure]]"
      - "[[knowledge_base/mathematics/softmax_function]]"
      - "[[knowledge_base/mathematics/numerical_stability]]"
---

# Matrix Operations

## Overview
Discrete active inference reduces to a few matrix operations on stochastic matrices: multiplying a belief vector by a column-stochastic matrix to predict, normalising columns to restore probabilities, and taking entropies and divergences of the results. The repository fixes one convention, columns sum to one, and implements the operations in `models/matrices/matrix_ops.py` and `utils/matrix_utils.py`.

## Definition / Formalism
A matrix $M\in\mathbb R^{m\times n}_{\ge0}$ is column-stochastic if $\sum_iM_{ij}=1$ for every column $j$. Then $Mq$ maps a distribution over $n$ outcomes to one over $m$, preserving normalisation. In the discrete model:

- Likelihood $A_{os}=P(o\mid s)$: predicted observations $q(o)=A\,q(s)$.
- Transitions $B_{s's}^{(a)}=P(s'\mid s,a)$, one column-stochastic slice per action: $q(s')=B^{(a)}q(s)$.

Worked check with $A=\begin{pmatrix}0.9&0.1\\0.1&0.9\end{pmatrix}$, $q(s)=(0.7,0.3)$: action 0 (identity $B$) leaves $q(s')=(0.7,0.3)$ and gives $q(o)=(0.66,0.34)$; action 1 (swap-like $B^{(1)}=\begin{pmatrix}0.1&0.9\\0.9&0.1\end{pmatrix}$) gives $q(s')=(0.34,0.66)$ and $q(o)=(0.372,0.628)$.

Column normalisation divides each column by its sum. `MatrixOps.normalize_columns` first clamps negatives to zero, and a column with zero mass becomes uniform instead of producing a division by zero. `normalize_rows` does the same along rows of a two-dimensional matrix.

Other operations provided: Shannon entropy (natural log) with a sum-to-one check, `compute_kl_divergence` with shape and normalisation checks, and a temperature-scaled stable softmax $\mathrm{softmax}(x/T)$. `MatrixInitializer` offers `random_stochastic`, `identity_based(shape, strength=0.9)` (diagonal $\text{strength}$, off-diagonals sharing the rest) and `uniform`. `MatrixLoader` reads YAML front matter specifications and checks shape and constraints such as `sum(cols) == 1.0`.

Two caveats: `utils.matrix_utils.ensure_matrix_properties` accepts `"column_stochastic"` or `"row_stochastic"` but raises an error if both are requested, so the orientation must be chosen; and `uniform(shape)` normalises over the whole array, so it is a uniform joint, not column-stochastic.

## Role in Active Inference
These routines are the numerical core of prediction and of expected free energy. `utils.matrix_utils.expected_free_energy` builds $A$ and a selected slice of $B$, predicts $q(s')$ and $q(o)$, takes risk as $D_{\mathrm{KL}}(q(o)\,\|\,\mathrm{softmax}(C))$, and ambiguity as the $q(s')$-weighted entropy of the columns of $A$.

## Related concepts
- [[linear_algebra]] — vector spaces, products and norms behind these operations
- [[pomdp_structure]] — where $A$, $B$, $C$, $D$ come from
- [[softmax_function]] — turning log-preferences into a distribution
- [[numerical_stability]] — max-shift and clipping in softmax
- [[random_variables]] — each column is a probability mass function
- [[expected_free_energy]] — computed with these primitives
