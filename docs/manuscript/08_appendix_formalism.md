# Appendix: Formalism and API correspondence {#sec:appendix_formalism}

## Probability invariants

Let $v\in\mathbb{R}^n$ be a probability vector. The normalized runtime
contract is

$$
v_i\geq 0,\qquad \sum_{i=1}^{n}v_i=1,
$$ {#eq:normalization_invariant}

up to floating-point tolerance. `compute_entropy`, `kl_divergence`, and the
dispatcher's private normalization boundary reject malformed inputs or return
a normalized result. For an N-dimensional tensor, the chosen axis is part of
the operation, so a caller can distinguish row, column, and final-axis
normalization.

## Policy enumeration

For $m$ actions and horizon $H$, the finite policy set is

$$
\Pi_H = \{0,\ldots,m-1\}^{H},\qquad |\Pi_H|=m^H,
$$ {#eq:policy_enumeration}

subject to the configured `policy_limit`. The dispatcher groups policies by
their first action, evaluates each sequence with discount factor $\gamma$,
and applies the policy distribution in [@eq:policy_distribution]. The limit
prevents accidental combinatorial growth from becoming an implicit runtime
failure.

## API map

| Mathematical object | Public implementation |
| --- | --- |
| $A,B,C,D,E$ | `cognitive.DiscreteGenerativeModel` |
| $q(s\mid o)$ | `DiscreteGenerativeModel.posterior` |
| $G(\pi)$ | `DiscreteGenerativeModel.evaluate_policy` |
| $q(a)$ | `ActiveInferenceDispatcher.dispatch_policy_inference` |
| $\Pi_y$ and $\Pi_x$ | `ContinuousActiveInference` precision vectors |
| persisted $S$ | `ActiveInferenceModel.save_state` / `load_state` |
| matrix constraints | `cognitive.utils.matrix_utils` |

## Error policy

Invalid state, action, observation, temperature, precision, matrix shape,
configuration key, or file path produces a diagnostic `ValueError` or
`TypeError`. The code does not silently select an unrelated mode when a
caller asks for unsupported behavior. This makes the public API narrower but
the behavior testable.
