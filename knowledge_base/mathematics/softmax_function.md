---

title: Softmax Function

type: mathematical_concept

status: stable

created: 2025-08-08

tags: [probability, optimization, normalization]

semantic_relations:

  - type: used_by

    links: [policy_selection, action_distribution]

  - type: relates

    links: [temperature_parameter, precision_parameter, numerical_stability]

---

## Softmax Function

### Definition

```math

\operatorname{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}

```

With temperature \(T\) or precision \(\gamma=1/T\): \(\operatorname{softmax}(\gamma x)\).

### Stability tips

- Use \(x \leftarrow x - \max_i x_i\)

- Clip inputs/\(\gamma\) to avoid overflow

### See also

- [[policy_selection]] · [[action_distribution]] · [[precision_parameter]] · [[temperature_parameter]] · [[numerical_stability]]

