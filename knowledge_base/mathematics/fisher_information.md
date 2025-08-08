---

title: Fisher Information

type: mathematical_concept

status: stable

created: 2025-08-08

tags: [information-theory, estimation, geometry]

semantic_relations:

  - type: relates

    links: [information_geometry, natural_gradients]

  - type: supports

    links: [policy_selection, action_distribution]

---

## Fisher Information

### Definition

Parameteric family \( p(x\mid\theta) \). Fisher information matrix:

```math

I(\theta) = \mathbb{E}_{p(x\mid\theta)}\left[ \nabla_\theta \log p(x\mid\theta)\, \nabla_\theta \log p(x\mid\theta)^\top \right]

```

### Roles

- [[information_geometry]]: induces the Fisher–Rao metric on statistical manifolds

- Natural gradient scaling in [[natural_gradients]]

- Sensitivity of likelihood; Cramér–Rao bounds

### Notes

- Estimate via Monte Carlo or analytic expectations

- Regularize for numerical stability (e.g., damping)

### See also

- [[information_geometry]] · [[natural_gradients]] · [[policy_selection]]

