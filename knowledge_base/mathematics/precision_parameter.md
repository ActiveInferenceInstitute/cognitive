---

title: Precision Parameter (Inverse Temperature)

type: mathematical_concept

status: stable

created: 2025-08-08

tags: [active-inference, control, temperature]

aliases: [inverse_temperature, gamma]

semantic_relations:

  - type: relates

    links: [temperature_parameter, policy_selection, action_distribution]

---

## Precision Parameter (γ)

The precision parameter γ controls the sharpness of the softmax used to form posteriors over policies and to sample actions in active inference. Higher γ yields more deterministic, exploitative selection; lower γ induces more stochastic, exploratory behavior.

Equivalently, γ is the inverse temperature: see [[temperature_parameter]] with \(\gamma = 1/T\).

### Mathematical definition

Policy posterior under expected free energy:

```math

q(\pi) = \operatorname{softmax}(-\gamma\, G(\pi))\quad\text{with}\quad

\operatorname{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}

```

Action posterior (single-step selection) obeys the same form with an action-value surrogate (e.g., expected free energy, negative cost, or log preference):

```math

q(a\,|\,s) = \operatorname{softmax}(\gamma\, U(s,a))

```

Relationship to temperature T: see [[temperature_parameter]] with \(\gamma = 1/T\). As \(T\to 0\) (\(\gamma\to\infty\)), selection becomes argmax; as \(T\to\infty\) (\(\gamma\to 0\)), selection approaches uniform.

### Roles in active inference

- Exploration–exploitation control: tunes the relative weight of small differences in \(G(\pi)\).

- Robustness to model error: lower \(\gamma\) hedges against misspecification by spreading probability mass.

- Precision weighting consistency: complements sensory and state precisions that scale prediction errors in variational updates.

### Estimating and adapting γ

1. Fixed hyperparameter: choose via validation to meet task risk or entropy targets.

1. Entropy-targeted adaptation: adjust \(\gamma\) to track a desired posterior entropy \(H[q(\pi)]\).

1. Uncertainty coupling: set \(\gamma_t = f(\hat{\sigma}_t)\) decreasing in epistemic uncertainty; explore when uncertain, exploit when confident.

1. Gradient tuning: treat \(\gamma\) as a differentiable scalar trained to minimize cumulative free energy (with regularization to avoid collapse).

Example entropy controller:

```math

\gamma_{t+1} = \gamma_t + \eta\,\big(H^* - H[q_t(\pi)]\big)\,,\quad \gamma\in[\gamma_{\min},\gamma_{\max}]

```

### Numerical stability

- Use stabilized softmax: subtract \(\max_i(-\gamma G_i)\) before exponentiation.

- Cap \(\gamma\) to avoid overflow; typical ranges: \(10^{-2}\)–\(10^2\) depending on the scale of \(G\).

- Normalize or standardize \(G(\pi)\) across batches to keep effective temperature in a stable regime.

### Policy vs sensory/state precisions

Active inference uses multiple precisions: policy precision γ (decision determinism) and sensory/state precisions Π (weighting of prediction errors). See [[../cognitive/precision_weighting]] for perceptual precisions and how they interact with γ to determine exploration and confidence.

### Reference implementation

```python

import numpy as np

def policy_posterior_from_efe(expected_free_energy: np.ndarray,

                              gamma: float,

                              min_prob: float = 1e-12) -> np.ndarray:

    """Softmax over negative EFE with inverse temperature gamma.

    Args:

        expected_free_energy: array-like of G(pi) values

        gamma: inverse temperature (precision)

        min_prob: floor to avoid exact zeros

    Returns:

        Probabilities over policies

    """

    logits = -gamma * expected_free_energy

    logits = logits - np.max(logits)

    probs = np.exp(logits)

    probs = np.maximum(probs, min_prob)

    return probs / probs.sum()

```

### Connections

- [[../mathematics/policy_selection]]: posterior formation over policies via softmax.

- [[../mathematics/action_distribution]]: action sampling with inverse temperature.

- [[../mathematics/exploration_exploitation]]: behavioral trade-off induced by \(\gamma\).

- [[../mathematics/expected_free_energy]] and [[../mathematics/vfe_components]]: scaling differences in value signals.

- [[../cognitive/active_inference]]: end-to-end framework context

- [[temperature_parameter]]: direct mapping to temperature T

## See also

- [[../cognitive/precision_weighting]]

- [[../mathematics/temperature_parameter]]

- [[../../docs/guides/learning_paths/active_inference_security_learning_path|Security Learning Path: precision bounds]]

### Worked example

Suppose three candidate policies have expected free energies: \(G = [0.0,\, 0.4,\, 1.0]\). The softmax over \(-\gamma G\) yields:

```math

\gamma = 0.5 \Rightarrow q(\pi) \approx [0.43,\, 0.36,\, 0.21] \\

\gamma = 2.0 \Rightarrow q(\pi) \approx [0.62,\, 0.25,\, 0.13]

```

Increasing \(\gamma\) concentrates mass on the lowest-\(G\) policy.

### Implementation snippet (Python)

```python

import numpy as np

def softmax_logit(values: np.ndarray) -> np.ndarray:

    z = values - np.max(values)

    e = np.exp(z)

    return e / e.sum()

def policy_posterior_from_efe(G: np.ndarray, gamma: float) -> np.ndarray:

    return softmax_logit(-gamma * G)

```

### References and further reading

- Softmax temperature control in probabilistic decision making

- Precision weighting in predictive coding and active inference (links: [[../cognitive/precision_weighting]], [[../cognitive/predictive_coding]])

