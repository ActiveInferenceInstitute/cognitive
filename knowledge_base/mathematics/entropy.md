---

title: Entropy

type: mathematical_concept

status: stable

created: 2025-08-08

tags: [information-theory, uncertainty]

semantic_relations:

  - type: relates

    links: [information_theory, mutual_information, kl_divergence]

  - type: used_by

    links: [expected_free_energy, action_distribution, exploration_exploitation]

---

## Entropy

### Shannon entropy

```math

H[X] = - \sum_x p(x)\,\log p(x) \quad (\text{discrete})

```

### Conditional and joint

```math

H[X\mid Y] = H[X,Y] - H[Y]\,,\quad H[X,Y] = -\sum_{x,y} p(x,y)\log p(x,y)

```

### Roles

- Uncertainty measure in exploration: higher H encourages information-seeking

- Appears in EFE via ambiguity terms and expected entropy reductions

### See also

- [[information_theory]] · [[mutual_information]] · [[kl_divergence]]

