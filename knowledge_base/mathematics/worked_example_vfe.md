---

title: Worked Example — Variational Free Energy

type: example

status: draft

created: 2025-08-08

tags: [example, vfe]

semantic_relations:

  - type: demonstrates

    links: [variational_free_energy]

---

## VFE on a simple categorical model

### Setup

Observation model \(p(o\mid s)\), prior \(p(s)\); recognition \(q(s)\).

### Compute

```math

F[q] = \mathbb{E}_{q(s)}[\log q(s) - \log p(o,s)]

```

Walk-through shows accuracy and complexity terms and gradient for \(q\).

