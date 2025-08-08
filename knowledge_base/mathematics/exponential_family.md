---

title: Exponential Family Distributions

type: mathematical_concept

status: stable

created: 2025-08-08

tags: [probability, statistics]

semantic_relations:

  - type: relates

    links: [variational_inference, fisher_information, natural_gradients]

  - type: used_by

    links: [message_passing, belief_updating]

---

## Exponential Families

### Canonical form

```math

p(x\mid\eta) = h(x) \exp\big( \eta^\top T(x) - A(\eta) \big)

```

- Natural parameter \(\eta\), sufficient statistics \(T(x)\), log-partition \(A(\eta)\)

### Properties

- Conjugacy simplifies updates in [[variational_inference]] and [[message_passing]]

- [[fisher_information]] equals Hessian of \(A(\eta)\)

### See also

- [[variational_inference]] · [[message_passing]] · [[natural_gradients]]

