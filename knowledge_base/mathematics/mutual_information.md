---

title: Mutual Information

type: mathematical_concept

status: stable

created: 2025-08-08

tags: [information-theory]

semantic_relations:

  - type: relates

    links: [entropy, kl_divergence]

  - type: used_by

    links: [expected_free_energy, epistemic_value]

---

## Mutual Information

### Definition

```math

I(X;Y) = D_{\mathrm{KL}}\big(p(x,y)\,\|\,p(x)p(y)\big) = H[X] + H[Y] - H[X,Y]

```

### Roles

- Quantifies information gained about states from observations

- Proxy for epistemic value and exploration objectives

### See also

- [[entropy]] · [[kl_divergence]] · [[expected_free_energy]]

