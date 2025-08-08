---

title: Bethe Free Energy

type: mathematical_concept

status: stable

created: 2025-08-08

tags: [variational, graphical-models]

semantic_relations:

  - type: relates

    links: [message_passing, belief_propagation]

  - type: connects

    links: [variational_free_energy]

---

## Bethe Free Energy

### Definition (factor graphs)

Let \(b_i(x_i)\) and \(b_a(\mathbf{x}_a)\) be variable and factor beliefs. The Bethe functional:

```math

F_{\mathrm{Bethe}}(b) = \sum_a \sum_{\mathbf{x}_a} b_a(\mathbf{x}_a) \log \frac{b_a(\mathbf{x}_a)}{\psi_a(\mathbf{x}_a)} - \sum_i (d_i-1) \sum_{x_i} b_i(x_i) \log b_i(x_i)

```

where \(d_i\) is the degree of variable \(i\). Stationary points under consistency constraints correspond to (loopy) BP fixed points.

### Notes

- Exact on trees; approximate on loopy graphs

- Links variational stationary conditions to message passing updates

### See also

- [[message_passing]] · [[belief_propagation]] · [[variational_free_energy]]

