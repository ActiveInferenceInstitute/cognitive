---

title: Variational Bounds and ELBO

type: mathematical_concept

status: stable

created: 2025-08-08

tags: [variational-inference]

semantic_relations:

  - type: relates

    links: [variational_inference, variational_free_energy]

  - type: used_by

    links: [message_passing, belief_updating]

---

## Variational Bounds

### ELBO

```math

\log p(o) \ge \mathbb{E}_{q(s)}[\log p(o,s)] - \mathbb{E}_{q(s)}[\log q(s)] \equiv -F[q]

```

### Notes

- Tight when \(q(s)=p(s\mid o)\)

- Connects to [[variational_free_energy]] decomposition into accuracy and complexity

### See also

- [[variational_inference]] · [[variational_free_energy]]

